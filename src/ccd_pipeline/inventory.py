"""Inventory raw frames for a night.

Reads primary headers only (fast) to tally ``OBSTYPE``, filters, and objects
before any calibration is run.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from astropy.io import fits

from .calibrate import exposure_seconds
from .io import filter_key, iter_fits_files, resolve_filter_name

# Always drop frames whose OBJECT matches these (case-insensitive substring),
# even if a night YAML omits them. "junk" is a common HDI discard label.
DEFAULT_EXCLUDE_OBJECT_SUBSTRINGS = ("junk",)


def excluded_object_substrings(cfg: dict) -> list[str]:
    """Substrings that mark an OBJECT header as unusable.

    Merges ``science.exclude_object_substrings`` with
    ``DEFAULT_EXCLUDE_OBJECT_SUBSTRINGS`` (always includes ``junk``).
    """
    configured = list((cfg.get("science") or {}).get("exclude_object_substrings") or [])
    merged: list[str] = []
    seen: set[str] = set()
    for item in list(configured) + list(DEFAULT_EXCLUDE_OBJECT_SUBSTRINGS):
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(text)
    return merged


def is_excluded_object(obj: str, cfg: dict) -> bool:
    """True if ``OBJECT`` matches any exclude substring (e.g. ``junk``)."""
    text = (obj or "").lower()
    if not text:
        return False
    return any(sub.lower() in text for sub in excluded_object_substrings(cfg))


def should_skip_science_object(obj: str, cfg: dict) -> bool:
    """Science-frame filter: include list (if set) + exclude substrings."""
    sci = cfg.get("science") or {}
    include = sci.get("include_objects") or []
    if include and obj not in include:
        return True
    return is_excluded_object(obj, cfg)


def excluded_files(cfg: dict) -> set[str]:
    """File names listed in ``exclude_files`` (frame QC rejects), by basename."""
    raw = cfg.get("exclude_files") or []
    if isinstance(raw, str):
        raw = [raw]
    return {Path(str(item)).name for item in raw if str(item).strip()}


def is_excluded_file(file_name: str, cfg: dict) -> bool:
    """True when the raw frame was rejected by name (``exclude_files``)."""
    return Path(str(file_name)).name in excluded_files(cfg)


def is_flat_obstype(value: str, configured: str | None = None) -> bool:
    """True for dome/sky/twilight flats, including ``SKYFLAT`` / ``DOMEFLAT``.

    Some cameras write ``FLAT``; Lowell Hall writes ``SKYFLAT``. Matching only
    the configured string would skip every flat on those nights.
    """
    got = str(value or "").strip().upper()
    if not got:
        return False
    want = str(configured or "").strip().upper()
    if want and got == want:
        return True
    return "FLAT" in got


def filter_excluded_rows(
    rows: list[dict],
    cfg: dict,
    *,
    warn: bool = True,
) -> list[dict]:
    """Drop rows whose OBJECT or file name is excluded; print what was skipped."""
    kept: list[dict] = []
    skipped_object: list[dict] = []
    skipped_file: list[dict] = []
    for row in rows:
        if is_excluded_file(row.get("file") or "", cfg):
            skipped_file.append(row)
        elif is_excluded_object(row.get("object") or "", cfg):
            skipped_object.append(row)
        else:
            kept.append(row)
    if warn and (skipped_object or skipped_file):
        from .term import dim, warn as warn_msg

        if skipped_object:
            subs = ", ".join(repr(s) for s in excluded_object_substrings(cfg))
            warn_msg(
                f"Skipping {len(skipped_object)} frame(s) with excluded OBJECT "
                f"(substrings: {subs}):"
            )
            for row in skipped_object:
                print(
                    dim(
                        f"  skip {row.get('file')}  "
                        f"OBJECT={row.get('object')!r}  OBSTYPE={row.get('obstype')}"
                    )
                )
        if skipped_file:
            warn_msg(
                f"Skipping {len(skipped_file)} frame(s) listed in exclude_files (frame QC):"
            )
            for row in skipped_file:
                print(dim(f"  skip {row.get('file')}  OBSTYPE={row.get('obstype')}"))
    return kept


def iter_fits(raw_dir: Path):
    yield from iter_fits_files(raw_dir)


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
        # EXPTIME, else DARKTIME (KPNO darks have EXPTIME 0), else the raw card
        exptime = exposure_seconds(hdr, kw["exptime"])
        if exptime is None:
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
        if is_flat_obstype(obstype, obstypes.get("flat")):
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
