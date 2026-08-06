"""Inventory raw frames for a night.

Reads primary headers only (fast) to tally ``OBSTYPE``, filters, and objects
before any calibration is run.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from astropy.io import fits

from .io import filter_key, resolve_filter_name


def iter_fits(raw_dir: Path):
    for path in sorted(Path(raw_dir).glob("*.fits")):
        if path.name.lower() == "latest.fits":
            continue
        yield path


def inventory_night(cfg: dict) -> dict:
    raw_dir = Path(cfg["paths"]["raw_dir"])
    kw = cfg["keywords"]
    obstypes = cfg["obstypes"]
    filter_map = cfg.get("filter_map", {})

    rows = []
    by_type = Counter()
    flats_by_filter = Counter()
    science_by_object = Counter()
    science_by_filter = Counter()

    for path in iter_fits(raw_dir):
        hdr = fits.getheader(path, 0)
        obstype = str(hdr.get(kw["obstype"], "")).strip().upper()
        exptime = hdr.get(kw["exptime"])
        obj = str(hdr.get(kw["object"], "")).strip()
        fkey = filter_key(hdr, kw["filter1"], kw["filter2"])
        fname = resolve_filter_name(hdr, filter_map, kw)

        rows.append(
            {
                "file": path.name,
                "path": str(path),
                "obstype": obstype,
                "exptime": exptime,
                "object": obj,
                "filter_key": fkey,
                "filter": fname,
            }
        )
        by_type[obstype] += 1
        if obstype == obstypes["flat"].upper():
            flats_by_filter[fname] += 1
        if obstype == obstypes["science"].upper():
            science_by_object[obj] += 1
            science_by_filter[fname] += 1

    return {
        "raw_dir": str(raw_dir),
        "n_files": len(rows),
        "by_type": dict(by_type),
        "flats_by_filter": dict(flats_by_filter),
        "science_by_object": dict(science_by_object),
        "science_by_filter": dict(science_by_filter),
        "rows": rows,
    }


def format_inventory(summary: dict) -> str:
    from .term import S, style

    lines = [
        f"Raw directory: {summary['raw_dir']}",
        f"Total FITS (excluding latest.fits): {summary['n_files']}",
        "",
        style("By OBSTYPE:", S.BOLD, S.CYAN),
    ]
    for k, n in sorted(summary["by_type"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {n:4d}  {k}")

    lines += ["", style("Flats by filter:", S.BOLD, S.CYAN)]
    for k, n in sorted(summary["flats_by_filter"].items()):
        lines.append(f"  {n:4d}  {k}")

    lines += ["", style("Science by filter:", S.BOLD, S.CYAN)]
    for k, n in sorted(summary["science_by_filter"].items()):
        lines.append(f"  {n:4d}  {k}")

    lines += ["", style("Science by OBJECT (top 30):", S.BOLD, S.CYAN)]
    items = sorted(summary["science_by_object"].items(), key=lambda kv: (-kv[1], kv[0]))
    for k, n in items[:30]:
        lines.append(f"  {n:4d}  {k}")
    return "\n".join(lines)
