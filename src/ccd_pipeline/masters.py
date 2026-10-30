"""Build master bias and master flats.

Follows the Astropy CCD Reduction Guide (see README Methodology):

* Overscan+trim each bias, then combine — GUIDE['bias_calibrate'], GUIDE['bias_combine']
  https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/02-02-Calibrating-bias-images.html
  https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/02-04-Combine-bias-images-to-make-master.html
* Flats: overscan, trim, subtract master bias, inv-median combine —
  https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-03-Calibrating-the-flats.html
  https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-04-Combining-flats.html
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median

from .calibrate import (
    apply_overscan_and_trim,
    combine_frames,
    darks_enabled,
    exposure_seconds,
    inv_median,
    print_processing_plan,
    subtract_bias,
    subtract_dark_current,
)
from .config import ensure_dirs
from .inventory import filter_excluded_rows, inventory_night, is_flat_obstype
from .io import master_flat_path, read_ccd, write_ccd
from .term import (
    info,
    progress_done,
    progress_status,
    run_with_spinner,
    subheading,
    success,
    warn,
)


def master_dark_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["masters_dir"]) / "master_dark.fits"


def load_master_dark(cfg: dict):
    """Load ``master_dark.fits`` when this night subtracts darks."""
    if not darks_enabled(cfg):
        return None
    path = master_dark_path(cfg)
    if not path.exists():
        raise RuntimeError(
            f"Master dark not found: {path}. Run the dark step before flats and science."
        )
    return read_ccd(path, image_hdu=0, unit=cfg["fits"]["unit"])


def _rows_of_type(summary: dict, obstype: str, *, kind: str | None = None) -> list[dict]:
    want = obstype.upper()
    if kind == "flat" or "FLAT" in want:
        return [r for r in summary["rows"] if is_flat_obstype(r["obstype"], obstype)]
    return [r for r in summary["rows"] if r["obstype"] == want]


def make_master_bias(cfg: dict) -> Path:
    """Calibrate bias frames and combine into ``master_bias.fits``.

    Guide 02-02 / 02-04: overscan+trim each bias, then σ-clip average combine.
    Master holds residual 2-D structure relative to overscan (median ≈ 0).
    """
    ensure_dirs(cfg)
    summary = inventory_night(cfg)
    bias_rows = filter_excluded_rows(
        _rows_of_type(summary, cfg["obstypes"]["bias"]), cfg
    )
    if not bias_rows:
        raise RuntimeError(
            "No bias frames found (after excluding OBJECT junk / filter list)"
        )

    print_processing_plan(cfg, kind="bias", title="Master bias")
    n = len(bias_rows)
    info(f"{n} bias frame(s)")

    calibrated = []
    for i, row in enumerate(bias_rows, start=1):
        progress_status(f"bias [{i}/{n}]  {row['file']}  overscan+trim")
        ccd = read_ccd(
            row["path"],
            image_hdu=cfg["fits"]["image_hdu"],
            unit=cfg["fits"]["unit"],
        )
        # Guide 02-02: overscan+trim BEFORE combining biases
        calibrated.append(apply_overscan_and_trim(ccd, cfg, verbose=False))

    progress_done(f"calibrated {n} bias frame(s)")

    comb = cfg["combine"]["bias"]
    # Guide 02-04: σ-clip average of overscan-calibrated biases → master bias
    master = run_with_spinner(
        "combine calibrated biases",
        combine_frames,
        calibrated,
        method=comb.get("method", "average"),
        sigma_clip=comb.get("sigma_clip", True),
        sigma_clip_low=comb.get("sigma_clip_low", 5),
        sigma_clip_high=comb.get("sigma_clip_high", 5),
        mem_limit=comb.get("mem_limit"),
        verbose=False,
    )
    master.meta["IMAGETYP"] = "BIAS"
    master.meta["OBSTYPE"] = "BIAS"
    master.meta["NCOMBINE"] = len(calibrated)

    out = Path(cfg["paths"]["masters_dir"]) / "master_bias.fits"
    write_ccd(master, out)
    success(f"Wrote master bias: {out}  shape={master.shape}")
    return out


def _scale_dark_to(ccd, target_s: float, exptime_key: str):
    """Multiply a calibrated dark so its exposure matches ``target_s``."""
    current = exposure_seconds(ccd.meta, exptime_key)
    if current is None or abs(current - target_s) / target_s < 1e-6:
        return ccd
    scaled = ccd.multiply(target_s / current)
    scaled.meta[exptime_key] = target_s
    if exptime_key != "EXPTIME":
        scaled.meta["EXPTIME"] = target_s
    return scaled


def make_master_dark(cfg: dict) -> Path | None:
    """Overscan, trim, subtract bias, and combine darks into ``master_dark.fits``.

    Darks taken at different exposure times are scaled to their median exposure
    before the combine. Flats and science later scale this master to each
    frame's exposure. Returns ``None`` when darks are disabled or no dark
    frames exist; in the latter case dark subtraction is turned off for the
    rest of this run.
    """
    if not darks_enabled(cfg):
        info("Darks disabled — not building a master dark.")
        return None

    ensure_dirs(cfg)
    summary = inventory_night(cfg)
    dark_value = str((cfg.get("obstypes") or {}).get("dark", "DARK"))
    dark_rows = filter_excluded_rows(_rows_of_type(summary, dark_value), cfg)
    if not dark_rows:
        warn(
            "darks.enabled is true, but no dark frames were found. "
            "Continuing without dark subtraction."
        )
        cfg.setdefault("darks", {})["enabled"] = False
        return None

    master_bias_path = Path(cfg["paths"]["masters_dir"]) / "master_bias.fits"
    if not master_bias_path.exists():
        raise RuntimeError(
            f"Master bias not found: {master_bias_path}. Run bias step first."
        )
    master_bias = read_ccd(master_bias_path, image_hdu=0, unit=cfg["fits"]["unit"])
    exptime_key = str((cfg.get("keywords") or {}).get("exptime") or "EXPTIME")

    print_processing_plan(cfg, kind="dark", title="Master dark")
    n = len(dark_rows)
    info(f"{n} dark frame(s)")

    calibrated = []
    exposures: list[float] = []
    for i, row in enumerate(dark_rows, start=1):
        progress_status(f"dark [{i}/{n}]  {row['file']}  overscan+trim −bias")
        ccd = read_ccd(
            row["path"],
            image_hdu=cfg["fits"]["image_hdu"],
            unit=cfg["fits"]["unit"],
        )
        ccd = apply_overscan_and_trim(ccd, cfg, verbose=False)
        ccd = subtract_bias(ccd, master_bias, verbose=False)
        calibrated.append(ccd)
        exp = exposure_seconds(ccd.meta, exptime_key)
        if exp is None and row.get("exptime") not in (None, ""):
            try:
                exp = float(row["exptime"])
            except (TypeError, ValueError):
                exp = None
        if exp is not None and exp > 0:
            exposures.append(exp)
            ccd.meta[exptime_key] = exp

    if exposures:
        target = float(median(exposures))
        unique = sorted({round(value, 6) for value in exposures})
        if len(unique) > 1:
            info(f"Scaling darks to {target:g}s before combine ({len(unique)} exposure times).")
        calibrated = [_scale_dark_to(ccd, target, exptime_key) for ccd in calibrated]
    else:
        target = None
        warn("Dark frames have no exposure time — master dark will be subtracted unscaled.")

    progress_done(f"calibrated {n} dark frame(s)")

    comb = (cfg.get("combine") or {}).get("dark") or (cfg.get("combine") or {}).get("bias") or {}
    master = run_with_spinner(
        "combine calibrated darks",
        combine_frames,
        calibrated,
        method=comb.get("method", "average"),
        sigma_clip=comb.get("sigma_clip", True),
        sigma_clip_low=comb.get("sigma_clip_low", 5),
        sigma_clip_high=comb.get("sigma_clip_high", 5),
        mem_limit=comb.get("mem_limit"),
        verbose=False,
    )
    master.meta["IMAGETYP"] = "DARK"
    master.meta["OBSTYPE"] = "DARK"
    master.meta["NCOMBINE"] = len(calibrated)
    if target is not None:
        master.meta[exptime_key] = target
        master.meta["EXPTIME"] = target

    out = master_dark_path(cfg)
    write_ccd(master, out)
    success(f"Wrote master dark: {out}  shape={master.shape}")
    return out


def make_master_flats(cfg: dict, *, filters: list[str] | None = None) -> dict[str, Path]:
    """Calibrate flats per filter and write ``master_flat_<filter>.fits``.

    Guide 05-03 / 05-04: overscan → trim → subtract master bias → inv-median
    scale → σ-clip combine. Multiplicative only; not a fringe correction.
    """
    ensure_dirs(cfg)
    summary = inventory_night(cfg)
    flat_rows = filter_excluded_rows(
        _rows_of_type(summary, cfg["obstypes"]["flat"], kind="flat"), cfg
    )
    if not flat_rows:
        raise RuntimeError(
            "No flat frames found (after excluding OBJECT junk / filter list)"
        )

    master_bias_path = Path(cfg["paths"]["masters_dir"]) / "master_bias.fits"
    if not master_bias_path.exists():
        raise RuntimeError(
            f"Master bias not found: {master_bias_path}. Run bias step first."
        )
    master_bias = read_ccd(master_bias_path, image_hdu=0, unit=cfg["fits"]["unit"])
    master_dark = load_master_dark(cfg)
    exptime_key = str((cfg.get("keywords") or {}).get("exptime") or "EXPTIME")

    by_filter: dict[str, list] = defaultdict(list)
    for row in flat_rows:
        if filters is not None and row["filter"] not in filters:
            continue
        by_filter[row["filter"]].append(row)
    if filters is not None and not by_filter:
        raise RuntimeError(f"No flats matched filters={filters}")

    comb = cfg["combine"]["flat"]
    # Guide 05-04: inv-median scale so exposure/level differences do not dominate
    scale = inv_median if comb.get("scale") == "inv_median" else None
    outputs: dict[str, Path] = {}

    flat_flags: list[str] = []
    for filt_name in filters or []:
        flat_flags.extend(["--filter", filt_name])
    print_processing_plan(cfg, kind="flat", title="Master flats", extra=flat_flags)

    for filt, rows in sorted(by_filter.items()):
        n = len(rows)
        subheading(f"filter '{filt}' — {n} frames")
        calibrated = []
        for i, row in enumerate(rows, start=1):
            progress_status(
                f"flat '{filt}' [{i}/{n}]  {row['file']}  "
                + ("overscan+trim −bias −dark" if master_dark is not None else "overscan+trim −bias")
            )
            ccd = read_ccd(
                row["path"],
                image_hdu=cfg["fits"]["image_hdu"],
                unit=cfg["fits"]["unit"],
            )
            # Guide 05-03: overscan/trim, then subtract master bias before combine.
            # Dark, when present, is subtracted after bias and scaled by exposure.
            ccd = apply_overscan_and_trim(ccd, cfg, verbose=False)
            ccd = subtract_bias(ccd, master_bias, verbose=False)
            if master_dark is not None:
                ccd = subtract_dark_current(ccd, master_dark, exptime_key=exptime_key)
            calibrated.append(ccd)

        progress_done(f"calibrated {n} flat(s) for filter '{filt}'")

        master = run_with_spinner(
            f"combine flats (filter={filt})",
            combine_frames,
            calibrated,
            method=comb.get("method", "average"),
            scale=scale,
            sigma_clip=comb.get("sigma_clip", True),
            sigma_clip_low=comb.get("sigma_clip_low", 5),
            sigma_clip_high=comb.get("sigma_clip_high", 5),
            mem_limit=comb.get("mem_limit"),
            verbose=False,
        )
        master.meta["IMAGETYP"] = "FLAT"
        master.meta["OBSTYPE"] = "FLAT"
        master.meta["FILTER"] = filt
        master.meta["NCOMBINE"] = len(calibrated)

        out = master_flat_path(cfg["paths"]["masters_dir"], filt)
        write_ccd(master, out)
        outputs[filt] = out
        success(f"Wrote master flat: {out.name}  filter={filt!r}  shape={master.shape}")

    return outputs


def build_all_masters(cfg: dict, *, filters: list[str] | None = None) -> None:
    make_master_bias(cfg)
    if darks_enabled(cfg):
        make_master_dark(cfg)
    make_master_flats(cfg, filters=filters)
