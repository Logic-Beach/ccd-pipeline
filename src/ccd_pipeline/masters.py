"""Build master bias and master flats."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .calibrate import (
    apply_overscan_and_trim,
    combine_frames,
    inv_median,
    subtract_bias,
)
from .config import ensure_dirs
from .inventory import inventory_night
from .io import read_ccd, write_ccd


def _rows_of_type(summary: dict, obstype: str) -> list[dict]:
    return [r for r in summary["rows"] if r["obstype"] == obstype.upper()]


def make_master_bias(cfg: dict) -> Path:
    ensure_dirs(cfg)
    summary = inventory_night(cfg)
    bias_rows = _rows_of_type(summary, cfg["obstypes"]["bias"])
    if not bias_rows:
        raise RuntimeError("No bias frames found")

    print(f"Building master bias from {len(bias_rows)} frames...")
    calibrated = []
    for row in bias_rows:
        ccd = read_ccd(
            row["path"],
            image_hdu=cfg["fits"]["image_hdu"],
            unit=cfg["fits"]["unit"],
        )
        calibrated.append(apply_overscan_and_trim(ccd, cfg))

    comb = cfg["combine"]["bias"]
    master = combine_frames(
        calibrated,
        method=comb.get("method", "average"),
        sigma_clip=comb.get("sigma_clip", True),
        sigma_clip_low=comb.get("sigma_clip_low", 5),
        sigma_clip_high=comb.get("sigma_clip_high", 5),
        mem_limit=comb.get("mem_limit", 4e9),
    )
    master.meta["IMAGETYP"] = "BIAS"
    master.meta["OBSTYPE"] = "BIAS"
    master.meta["NCOMBINE"] = len(calibrated)

    out = Path(cfg["paths"]["masters_dir"]) / "master_bias.fits"
    write_ccd(master, out)
    print(f"Wrote {out}")
    return out


def make_master_flats(cfg: dict, *, filters: list[str] | None = None) -> dict[str, Path]:
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
    # Masters are written as single-HDU CCDData files
    master_bias = read_ccd(master_bias_path, image_hdu=0, unit=cfg["fits"]["unit"])

    by_filter: dict[str, list] = defaultdict(list)
    for row in flat_rows:
        if filters is not None and row["filter"] not in filters:
            continue
        by_filter[row["filter"]].append(row)
    if filters is not None and not by_filter:
        raise RuntimeError(f"No flats matched filters={filters}")

    comb = cfg["combine"]["flat"]
    scale = inv_median if comb.get("scale") == "inv_median" else None
    outputs: dict[str, Path] = {}

    for filt, rows in sorted(by_filter.items()):
        print(f"Building master flat '{filt}' from {len(rows)} frames...")
        calibrated = []
        for row in rows:
            ccd = read_ccd(
                row["path"],
                image_hdu=cfg["fits"]["image_hdu"],
                unit=cfg["fits"]["unit"],
            )
            ccd = apply_overscan_and_trim(ccd, cfg)
            ccd = subtract_bias(ccd, master_bias)
            calibrated.append(ccd)

        master = combine_frames(
            calibrated,
            method=comb.get("method", "average"),
            scale=scale,
            sigma_clip=comb.get("sigma_clip", True),
            sigma_clip_low=comb.get("sigma_clip_low", 5),
            sigma_clip_high=comb.get("sigma_clip_high", 5),
            mem_limit=comb.get("mem_limit", 4e9),
        )
        master.meta["IMAGETYP"] = "FLAT"
        master.meta["OBSTYPE"] = "FLAT"
        master.meta["FILTER"] = filt
        master.meta["NCOMBINE"] = len(calibrated)

        out = Path(cfg["paths"]["masters_dir"]) / f"master_flat_{filt}.fits"
        write_ccd(master, out)
        outputs[filt] = out
        print(f"Wrote {out}")

    return outputs


def build_all_masters(cfg: dict, *, filters: list[str] | None = None) -> None:
    make_master_bias(cfg)
    make_master_flats(cfg, filters=filters)
