"""Reduce science frames with master bias + flats.

Follows Astropy CCD Reduction Guide 06-00 / 06-03 (README Methodology):

    overscan → trim → subtract master bias → divide by master flat

https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/06-00-Reducing-science-images.html
https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/06-03-science-images-calibration-examples.html

When the night has dark frames, a master dark is subtracted after the bias
and scaled by exposure time. Nights without darks skip that step.

Fringe correction (additive; i/z) is a *later* optional step — see ``fringe.py``.
Do not fringe-correct before flat-fielding.

Science products are named ``{stages}-{night_id}.{object}.{filter}.fits``
(with a numeric suffix when the same object+filter appears more than once).
The prefix records what has been applied: ``p`` processed, ``f`` fringe
corrected, ``w`` WCS solved — e.g. ``p-…``, ``pf-…``, ``pw-…``, ``pfw-…``.
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
    subtract_dark_current,
)
from .masters import load_master_dark
from .config import ensure_dirs
from .inventory import inventory_night, is_excluded_file, should_skip_science_object
from .io import filter_name_from_master_flat_path, read_ccd, write_ccd
from .term import info, progress_done, progress_status, success, warn

# Keep alphanumerics, plus/minus, dots, underscores; collapse the rest to "_"
_UNSAFE = re.compile(r"[^\w.+\-]+", re.UNICODE)
_MULTI_UNDERSCORE = re.compile(r"_+")
# Leading stage tag: p, pf, pw, or pfw, then a hyphen.
_SCIENCE_STAGE_RE = re.compile(r"^(?P<tag>p(?:f)?(?:w)?)-", re.IGNORECASE)


def _should_skip_object(obj: str, cfg: dict) -> bool:
    """Apply science.include_objects / exclude_object_substrings (incl. junk)."""
    return should_skip_science_object(obj, cfg)


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


def science_stage_tag(*, processed: bool = False, fringe: bool = False, wcs: bool = False) -> str:
    """Stage letters in fixed order: processed, fringe, WCS."""
    letters = ""
    if processed:
        letters += "p"
    if fringe:
        letters += "f"
    if wcs:
        letters += "w"
    return letters


def split_science_stage_name(filename: str) -> tuple[str, str]:
    """Split ``pf-night.obj.r.fits`` → ``('pf', 'night.obj.r.fits')``.

    Names without a stage prefix return ``('', filename)``.
    """
    text = Path(filename).name
    match = _SCIENCE_STAGE_RE.match(text)
    if not match:
        return "", text
    return match.group("tag").lower(), text[match.end() :]


def with_science_stages(
    path: Path | str,
    *,
    processed: bool | None = None,
    fringe: bool | None = None,
    wcs: bool | None = None,
) -> Path:
    """Return ``path`` with stage letters set (``None`` keeps the current flag)."""
    path = Path(path)
    tag, rest = split_science_stage_name(path.name)
    stages = set(tag)

    def _apply(flag: bool | None, letter: str) -> None:
        if flag is True:
            stages.add(letter)
        elif flag is False:
            stages.discard(letter)

    _apply(processed, "p")
    _apply(fringe, "f")
    _apply(wcs, "w")
    if "f" in stages or "w" in stages:
        stages.add("p")
    new_tag = science_stage_tag(
        processed="p" in stages, fringe="f" in stages, wcs="w" in stages
    )
    prefix = f"{new_tag}-" if new_tag else ""
    return path.with_name(prefix + rest)


def remove_unprefixed_science_twin(path: Path) -> Path | None:
    """Delete the legacy ``night.obj.r.fits`` sibling of ``p-night.obj.r.fits``."""
    path = Path(path)
    _, rest = split_science_stage_name(path.name)
    twin = path.with_name(rest)
    if twin.resolve() == path.resolve() or not twin.exists():
        return None
    twin.unlink()
    return twin


def retitle_science_file(
    path: Path,
    *,
    processed: bool | None = None,
    fringe: bool | None = None,
    wcs: bool | None = None,
) -> Path:
    """Rename a science product to match its processing stages; update FILENAME."""
    dest = with_science_stages(path, processed=processed, fringe=fringe, wcs=wcs)
    if dest.resolve() == Path(path).resolve():
        return Path(path)
    if dest.exists():
        dest.unlink()
    Path(path).replace(dest)
    try:
        from astropy.io import fits

        with fits.open(dest, mode="update", memmap=False) as hdul:
            hdul[0].header["FILENAME"] = dest.name
            hdul.flush()
    except Exception:
        pass
    return dest


def science_output_name(
    night_id: str,
    object_name: str,
    filter_name: str,
    *,
    occurrence: int,
    total_for_key: int,
) -> str:
    """Filename for one calibrated science product.

    Single exposure of that object+filter → ``p-2017JUN24.SA_103-Z.r.fits``.
    Repeats get a 1-based suffix → ``p-2017JUN24.SA_103-Z.r.01.fits``.
    Later fringe / WCS steps add ``f`` / ``w`` to the prefix.
    """
    seq = occurrence if total_for_key > 1 else None
    width = max(2, len(str(total_for_key)))
    stem = science_product_stem(
        night_id, object_name, filter_name, seq=seq, seq_width=width
    )
    return f"{science_stage_tag(processed=True)}-{stem}.fits"


def load_masters(cfg: dict) -> tuple[object, dict[str, object]]:
    masters_dir = Path(cfg["paths"]["masters_dir"])
    bias_path = masters_dir / "master_bias.fits"
    if not bias_path.exists():
        raise RuntimeError(f"Missing master bias: {bias_path}")

    master_bias = read_ccd(bias_path, image_hdu=0, unit=cfg["fits"]["unit"])
    flats = {}
    for path in sorted(masters_dir.glob("master_flat_*.fits")):
        ccd = read_ccd(path, image_hdu=0, unit=cfg["fits"]["unit"])
        # Prefer FILTER card so jc_I / sdss_i map correctly on case-insensitive disks.
        filt = filter_name_from_master_flat_path(path, ccd.meta)
        if filt in flats:
            warn(
                f"Duplicate master flat for filter {filt!r}: "
                f"{flats[filt].meta.get('ORIGFILE', '?')} and {path.name}"
            )
        flats[filt] = ccd
    if not flats:
        raise RuntimeError(f"No master flats found in {masters_dir}")
    return master_bias, flats


def reduce_science(
    cfg: dict,
    *,
    limit: int | None = None,
    filters: list[str] | None = None,
) -> list[Path]:
    """Calibrate science frames; write under ``<output_dir>/science/``."""
    ensure_dirs(cfg)
    summary = inventory_night(cfg)
    want = {f.strip() for f in (filters or []) if f and str(f).strip()}
    science_rows = [
        r
        for r in summary["rows"]
        if r["obstype"] == cfg["obstypes"]["science"].upper()
        and not _should_skip_object(r["object"], cfg)
        and not is_excluded_file(r["file"], cfg)
        and (not want or r["filter"] in want)
    ]

    master_bias, master_flats = load_masters(cfg)
    master_dark = load_master_dark(cfg)
    exptime_key = str((cfg.get("keywords") or {}).get("exptime") or "EXPTIME")
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
            warn(
                f"SKIP {row['file']}: no master flat for filter {row['filter']!r} "
                f"(have: {', '.join(sorted(master_flats))})"
            )
            continue
        planned_rows.append(row)
    totals: Counter[tuple[str, str]] = Counter(
        (r["object"], r["filter"]) for r in planned_rows
    )
    seen: dict[tuple[str, str], int] = defaultdict(int)

    reduce_flags: list[str] = []
    if limit is not None:
        reduce_flags.extend(["--limit", str(limit)])
    for filt_name in sorted(want):
        reduce_flags.extend(["--filter", filt_name])
    print_processing_plan(
        cfg, kind="science", title="Science calibration", extra=reduce_flags
    )
    info(f"Master bias : {cfg['paths']['masters_dir']}/master_bias.fits")
    if master_dark is not None:
        info(f"Master dark : {cfg['paths']['masters_dir']}/master_dark.fits")
    info(f"Master flats: {', '.join(sorted(master_flats))}")
    info(f"Output names: p-{night_id}.<OBJECT>.<filter>[.NN].fits")
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

        i = len(written) + 1
        progress_status(
            f"science [{i}/{n_planned}]  {row['file']} → {out_name}  "
            + (
                "overscan −bias −dark ÷flat"
                if master_dark is not None
                else "overscan −bias ÷flat"
            )
        )
        ccd = read_ccd(
            row["path"],
            image_hdu=cfg["fits"]["image_hdu"],
            unit=cfg["fits"]["unit"],
        )
        # Guide 06-00 / 06-03 — same order as flats, then flat_correct.
        # Fringe (additive) is NOT done here; run `ccd fringe` after reduce.
        ccd = apply_overscan_and_trim(ccd, cfg, verbose=False)
        ccd = subtract_bias(ccd, master_bias, verbose=False)
        if master_dark is not None:
            ccd = subtract_dark_current(ccd, master_dark, exptime_key=exptime_key)
        ccd = flat_correct(ccd, master_flats[filt], verbose=False)
        ccd.meta["REDUCED"] = True
        ccd.meta["FILTER"] = filt
        ccd.meta["FILENAME"] = out_name

        out = out_dir / out_name
        write_ccd(ccd, out)
        remove_unprefixed_science_twin(out)
        written.append(out)

    progress_done(f"wrote {len(written)} calibrated science frame(s)")
    success(f"Output: {out_dir}")
    return written
