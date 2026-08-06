"""Reduce science frames with master bias + flats.

Follows Astropy CCD Reduction Guide ``06-00-Reducing-science-images`` /
``06-03-science-images-calibration-examples``:

    overscan → trim → subtract master bias → divide by master flat

Dark subtraction is optional and disabled for nights without darks
(common for cryogenically cooled cameras with short exposures).

Guide: https://www.astropy.org/ccd-reduction-and-photometry-guide/
"""

from __future__ import annotations

from pathlib import Path

from .calibrate import (
    apply_overscan_and_trim,
    flat_correct,
    print_processing_plan,
    subtract_bias,
)
from .config import ensure_dirs
from .inventory import inventory_night
from .io import read_ccd, write_ccd
from .term import S, dim, info, style, success, warn


def _should_skip_object(obj: str, cfg: dict) -> bool:
    """Apply science.include_objects / exclude_object_substrings filters."""
    sci = cfg.get("science", {})
    include = sci.get("include_objects") or []
    if include and obj not in include:
        return True
    for substr in sci.get("exclude_object_substrings") or []:
        if substr.lower() in obj.lower():
            return True
    return False


def load_masters(cfg: dict) -> tuple[object, dict[str, object]]:
    masters_dir = Path(cfg["paths"]["masters_dir"])
    bias_path = masters_dir / "master_bias.fits"
    if not bias_path.exists():
        raise RuntimeError(f"Missing master bias: {bias_path}")

    master_bias = read_ccd(bias_path, image_hdu=0, unit=cfg["fits"]["unit"])
    flats = {}
    for path in sorted(masters_dir.glob("master_flat_*.fits")):
        filt = path.stem.replace("master_flat_", "", 1)
        flats[filt] = read_ccd(path, image_hdu=0, unit=cfg["fits"]["unit"])
    if not flats:
        raise RuntimeError(f"No master flats found in {masters_dir}")
    return master_bias, flats


def reduce_science(cfg: dict, *, limit: int | None = None) -> list[Path]:
    """Calibrate science frames; write under ``<output_dir>/science/``."""
    ensure_dirs(cfg)
    summary = inventory_night(cfg)
    science_rows = [
        r
        for r in summary["rows"]
        if r["obstype"] == cfg["obstypes"]["science"].upper()
        and not _should_skip_object(r["object"], cfg)
    ]

    master_bias, master_flats = load_masters(cfg)
    out_dir = Path(cfg["paths"]["output_dir"]) / "science"
    out_dir.mkdir(parents=True, exist_ok=True)

    print_processing_plan(cfg, kind="science", title="Science calibration")
    info(f"Master bias : {cfg['paths']['masters_dir']}/master_bias.fits")
    info(f"Master flats: {', '.join(sorted(master_flats))}")
    n_planned = len(science_rows) if limit is None else min(limit, len(science_rows))
    print(f"Science frames to process: {n_planned}")

    written: list[Path] = []
    for i, row in enumerate(science_rows):
        if limit is not None and i >= limit:
            break
        filt = row["filter"]
        if filt not in master_flats:
            warn(f"\nSKIP {row['file']}: no master flat for filter '{filt}'")
            continue

        print()
        print(
            style(f"[{len(written)+1}/{n_planned}] {row['file']}", S.BOLD, S.MAGENTA)
        )
        print(
            dim(
                f"    OBJECT={row['object']!r}  filter={filt}  EXPTIME={row['exptime']} s"
            )
        )
        ccd = read_ccd(
            row["path"],
            image_hdu=cfg["fits"]["image_hdu"],
            unit=cfg["fits"]["unit"],
        )
        print(dim(f"    loaded shape={ccd.shape}"))
        # Guide 06: same calibration chain as flats, then flat_correct
        ccd = apply_overscan_and_trim(ccd, cfg, verbose=True, label=row["file"])
        ccd = subtract_bias(ccd, master_bias, verbose=True, label=row["file"])
        ccd = flat_correct(ccd, master_flats[filt], verbose=True, label=row["file"])
        ccd.meta["REDUCED"] = True
        ccd.meta["FILTER"] = filt

        out = out_dir / row["file"]
        write_ccd(ccd, out)
        print(dim(f"    wrote {out}"))
        written.append(out)

    success(f"\nWrote {len(written)} calibrated science frames to {out_dir}")
    return written
