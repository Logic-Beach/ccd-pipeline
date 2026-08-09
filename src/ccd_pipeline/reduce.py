"""Reduce science frames with master bias + flats.

Follows Astropy CCD Reduction Guide ``06-00-Reducing-science-images`` /
``06-03-science-images-calibration-examples``:

    overscan → trim → subtract master bias → divide by master flat

Dark subtraction is optional and disabled for nights without darks
(common for cryogenically cooled cameras with short exposures).

Science products are named ``{night_id}.{object}.{filter}.fits`` (with a
numeric suffix when the same object+filter appears more than once).

Guide: https://www.astropy.org/ccd-reduction-and-photometry-guide/
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
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

# Keep alphanumerics, plus/minus, dots, underscores; collapse the rest to "_"
_UNSAFE = re.compile(r"[^\w.+\-]+", re.UNICODE)
_MULTI_UNDERSCORE = re.compile(r"_+")


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


def sanitize_filename_part(text: str, *, fallback: str = "unknown") -> str:
    """Make an OBJECT/filter string safe for a single filename component."""
    cleaned = _UNSAFE.sub("_", str(text).strip())
    cleaned = _MULTI_UNDERSCORE.sub("_", cleaned).strip("._")
    return cleaned or fallback


def science_product_stem(
    night_id: str,
    object_name: str,
    filter_name: str,
    *,
    seq: int | None = None,
    seq_width: int = 2,
) -> str:
    """Build ``{night}.{object}.{filter}`` or ``….{seq}`` stem (no extension)."""
    night = sanitize_filename_part(night_id, fallback="night")
    obj = sanitize_filename_part(object_name, fallback="object")
    filt = sanitize_filename_part(filter_name, fallback="filter")
    stem = f"{night}.{obj}.{filt}"
    if seq is not None:
        stem = f"{stem}.{seq:0{seq_width}d}"
    return stem


def science_output_name(
    night_id: str,
    object_name: str,
    filter_name: str,
    *,
    occurrence: int,
    total_for_key: int,
) -> str:
    """Filename for one science product.

    Single exposure of that object+filter → ``2017JUN24.SA_103-Z.r.fits``.
    Repeats get a 1-based suffix → ``2017JUN24.SA_103-Z.r.01.fits``.
    """
    seq = occurrence if total_for_key > 1 else None
    width = max(2, len(str(total_for_key)))
    return science_product_stem(
        night_id, object_name, filter_name, seq=seq, seq_width=width
    ) + ".fits"


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
    night_id = str(cfg.get("night_id") or Path(cfg["paths"]["raw_dir"]).name)

    # Pre-count object+filter occurrences among frames we will actually write
    # (respect --limit and missing flats) so suffixes are stable/complete.
    planned_rows: list[dict] = []
    for i, row in enumerate(science_rows):
        if limit is not None and i >= limit:
            break
        if row["filter"] not in master_flats:
            warn(f"SKIP {row['file']}: no master flat for filter '{row['filter']}'")
            continue
        planned_rows.append(row)
    totals: Counter[tuple[str, str]] = Counter(
        (r["object"], r["filter"]) for r in planned_rows
    )
    seen: dict[tuple[str, str], int] = defaultdict(int)

    print_processing_plan(cfg, kind="science", title="Science calibration")
    info(f"Master bias : {cfg['paths']['masters_dir']}/master_bias.fits")
    info(f"Master flats: {', '.join(sorted(master_flats))}")
    info(f"Output names: {night_id}.<OBJECT>.<filter>[.NN].fits")
    n_planned = len(planned_rows)
    print(f"Science frames to process: {n_planned}")

    written: list[Path] = []
    for row in planned_rows:
        filt = row["filter"]
        key = (row["object"], filt)
        seen[key] += 1
        out_name = science_output_name(
            night_id,
            row["object"],
            filt,
            occurrence=seen[key],
            total_for_key=totals[key],
        )

        print()
        print(
            style(
                f"[{len(written)+1}/{n_planned}] {row['file']} → {out_name}",
                S.BOLD,
                S.MAGENTA,
            )
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
        ccd.meta["FILENAME"] = out_name

        out = out_dir / out_name
        write_ccd(ccd, out)
        print(dim(f"    wrote {out}"))
        written.append(out)

    success(f"\nWrote {len(written)} calibrated science frames to {out_dir}")
    return written
