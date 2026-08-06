"""Reduce science frames with master bias + flats."""

from __future__ import annotations

from pathlib import Path

from .calibrate import apply_overscan_and_trim, flat_correct, subtract_bias
from .config import ensure_dirs
from .inventory import inventory_night
from .io import read_ccd, write_ccd


def _should_skip_object(obj: str, cfg: dict) -> bool:
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

    written: list[Path] = []
    for i, row in enumerate(science_rows):
        if limit is not None and i >= limit:
            break
        filt = row["filter"]
        if filt not in master_flats:
            print(f"SKIP {row['file']}: no master flat for filter '{filt}'")
            continue

        print(f"[{i+1}/{len(science_rows) if limit is None else limit}] {row['file']}  "
              f"{row['object']}  filter={filt}")
        ccd = read_ccd(
            row["path"],
            image_hdu=cfg["fits"]["image_hdu"],
            unit=cfg["fits"]["unit"],
        )
        ccd = apply_overscan_and_trim(ccd, cfg)
        ccd = subtract_bias(ccd, master_bias)
        ccd = flat_correct(ccd, master_flats[filt])
        ccd.meta["REDUCED"] = True
        ccd.meta["FILTER"] = filt

        out = out_dir / row["file"]
        write_ccd(ccd, out)
        written.append(out)

    print(f"Wrote {len(written)} calibrated science frames to {out_dir}")
    return written
