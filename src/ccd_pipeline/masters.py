"""Build master bias and master flats.

Follows the Astropy CCD Reduction Guide:

* ``02-02-Calibrating-bias-images`` — overscan + trim each bias
* ``02-04-Combine-bias-images-to-make-master`` — σ-clip combine → master bias
* ``05-03-Calibrating-the-flats`` — overscan, trim, subtract master bias
* ``05-04-Combining-flats`` — inverse-median scale + combine → master flat

Guide: https://www.astropy.org/ccd-reduction-and-photometry-guide/
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .calibrate import (
    apply_overscan_and_trim,
    combine_frames,
    inv_median,
    print_processing_plan,
    subtract_bias,
)
from .config import ensure_dirs
from .inventory import inventory_night
from .io import read_ccd, write_ccd
from .term import dim, step, subheading, success


def _rows_of_type(summary: dict, obstype: str) -> list[dict]:
    return [r for r in summary["rows"] if r["obstype"] == obstype.upper()]


def _print_file_list(title: str, rows: list[dict]) -> None:
    print(title)
    for row in rows:
        exp = row.get("exptime")
        obj = row.get("object") or ""
        filt = row.get("filter") or ""
        # EXPTIME is seconds (FITS). For biases it is often a controller
        # request time even when the shutter never opens — not sky exposure.
        extra = f"  EXPTIME={exp} s"
        if obj:
            extra += f"  OBJECT={obj!r}"
        if filt:
            extra += f"  filter={filt}"
        print(dim(f"  {row['file']}{extra}"))
    print(dim(f"  ({len(rows)} files)"))


def make_master_bias(cfg: dict) -> Path:
    """Calibrate bias frames and combine into ``master_bias.fits``."""
    ensure_dirs(cfg)
    summary = inventory_night(cfg)
    bias_rows = _rows_of_type(summary, cfg["obstypes"]["bias"])
    if not bias_rows:
        raise RuntimeError("No bias frames found")

    print_processing_plan(cfg, kind="bias", title="Master bias")
    _print_file_list(f"\nInput frames ({len(bias_rows)}):", bias_rows)

    calibrated = []
    for row in bias_rows:
        step(row["file"])
        ccd = read_ccd(
            row["path"],
            image_hdu=cfg["fits"]["image_hdu"],
            unit=cfg["fits"]["unit"],
        )
        print(dim(f"    loaded shape={ccd.shape}"))
        # Guide 02-02: overscan-subtract + trim BEFORE combining biases
        calibrated.append(
            apply_overscan_and_trim(ccd, cfg, verbose=True, label=row["file"])
        )

    comb = cfg["combine"]["bias"]
    step("combine calibrated biases")
    master = combine_frames(
        calibrated,
        method=comb.get("method", "average"),
        sigma_clip=comb.get("sigma_clip", True),
        sigma_clip_low=comb.get("sigma_clip_low", 5),
        sigma_clip_high=comb.get("sigma_clip_high", 5),
        mem_limit=comb.get("mem_limit"),
        verbose=True,
    )
    master.meta["IMAGETYP"] = "BIAS"
    master.meta["OBSTYPE"] = "BIAS"
    master.meta["NCOMBINE"] = len(calibrated)

    out = Path(cfg["paths"]["masters_dir"]) / "master_bias.fits"
    write_ccd(master, out)
    success(f"\nWrote master bias: {out}  shape={master.shape}")
    return out


def make_master_flats(cfg: dict, *, filters: list[str] | None = None) -> dict[str, Path]:
    """Calibrate flats per filter and write ``master_flat_<filter>.fits``."""
    ensure_dirs(cfg)
    summary = inventory_night(cfg)
    flat_rows = _rows_of_type(summary, cfg["obstypes"]["flat"])
    if not flat_rows:
        raise RuntimeError("No flat frames found")

    master_bias_path = Path(cfg["paths"]["masters_dir"]) / "master_bias.fits"
    if not master_bias_path.exists():
        raise RuntimeError(
            f"Master bias not found: {master_bias_path}. Run bias step first."
        )
    master_bias = read_ccd(master_bias_path, image_hdu=0, unit=cfg["fits"]["unit"])

    by_filter: dict[str, list] = defaultdict(list)
    for row in flat_rows:
        if filters is not None and row["filter"] not in filters:
            continue
        by_filter[row["filter"]].append(row)
    if filters is not None and not by_filter:
        raise RuntimeError(f"No flats matched filters={filters}")

    comb = cfg["combine"]["flat"]
    # Guide 05-04: scale each flat by 1/median so exposure/level differences
    # do not weight the stack unevenly; combined master then has median ≈ 1.
    scale = inv_median if comb.get("scale") == "inv_median" else None
    outputs: dict[str, Path] = {}

    print_processing_plan(cfg, kind="flat", title="Master flats")

    for filt, rows in sorted(by_filter.items()):
        subheading(f"filter '{filt}' — {len(rows)} frames")
        _print_file_list(f"Input frames ({len(rows)}):", rows)
        calibrated = []
        for row in rows:
            step(row["file"])
            ccd = read_ccd(
                row["path"],
                image_hdu=cfg["fits"]["image_hdu"],
                unit=cfg["fits"]["unit"],
            )
            print(dim(f"    loaded shape={ccd.shape}"))
            # Guide 05-03: overscan/trim, then subtract master bias (no dark here)
            ccd = apply_overscan_and_trim(ccd, cfg, verbose=True, label=row["file"])
            ccd = subtract_bias(ccd, master_bias, verbose=True, label=row["file"])
            calibrated.append(ccd)

        step(f"combine calibrated flats (filter={filt})")
        master = combine_frames(
            calibrated,
            method=comb.get("method", "average"),
            scale=scale,
            sigma_clip=comb.get("sigma_clip", True),
            sigma_clip_low=comb.get("sigma_clip_low", 5),
            sigma_clip_high=comb.get("sigma_clip_high", 5),
            mem_limit=comb.get("mem_limit"),
            verbose=True,
        )
        master.meta["IMAGETYP"] = "FLAT"
        master.meta["OBSTYPE"] = "FLAT"
        master.meta["FILTER"] = filt
        master.meta["NCOMBINE"] = len(calibrated)

        out = Path(cfg["paths"]["masters_dir"]) / f"master_flat_{filt}.fits"
        write_ccd(master, out)
        outputs[filt] = out
        success(f"\nWrote master flat: {out}  shape={master.shape}")

    return outputs


def build_all_masters(cfg: dict, *, filters: list[str] | None = None) -> None:
    make_master_bias(cfg)
    make_master_flats(cfg, filters=filters)
