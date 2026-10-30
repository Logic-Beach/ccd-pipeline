"""Build and apply additive fringe frames (typically SDSS z / i).

Fringing is thin-film interference in the CCD at discrete sky (airglow) lines.
It is an *additive* modulation of the sky background and is **not** removed by
dividing by broadband flats (dome/twilight flats are multiplicative and usually
show little fringe structure).

Astropy CCD Guide coverage
--------------------------
There is **no** fringe notebook in the Astropy CCD Reduction Guide. Flat
correction (multiplicative) is covered in 05-01 / 05-03 / 05-04; science
calibration stops at bias+flat (06-00). Fringe is a separate post-flat step —
see README “Methodology” §6.

Professional practice we follow
-------------------------------
First-order correction (IRAF ``mkfringecor`` / ``rmfringe``, NOIRLab SOI,
Astro-WISE fringe HOWTO):

1. Scale moonless night / blank-ish exposures by their median sky
2. Mask stars and median-stack → fringe pattern (sky ≈ 1 + ripple)
3. Zero-mean template ``F = S − median(S)``
4. Subtract ``sky_median × F`` from each science frame (after flat-fielding)

References (``FRINGE_REFS`` in ``__init__.py``):
* NOIRLab SOI — https://noirlab.edu/science/index.php/programs/ctio/instruments/soi/evaluating-and-reducing-soi-images
* IRAF mkfringecor — https://iraf.readthedocs.io/en/latest/tasks/noao/imred/ccdred/mkfringecor.html
* IRAF rmfringe — https://iraf.readthedocs.io/en/latest/tasks/nfextern/msctools/rmfringe.html
* Astro-WISE — https://doc.astro-wise.org/man_howto_fringe.html

Shape/amplitude still vary with the night-sky spectrum (Moon, time), so this is
approximate — enough to clean photometry of extended objects and cosmetics.
Sky-median scaling is the classic first-order scale; IRAF/LSST can fit the
amplitude more carefully when the spectrum changes strongly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats
from scipy.ndimage import binary_dilation

from . import FRINGE_REFS
from .reduce import remove_unprefixed_science_twin, retitle_science_file
from .science_qc import group_science_by_filter, parse_science_filename
from .term import dim, info, show_ccd, step, subheading, success, warn
from .wcs_solve import iter_science_fits, parse_pointing


DEFAULT_FRINGE = {
    # --- Which filters get fringe masters / apply ---
    # Short FILTER names (from filter_map / FILTER card), e.g. z, i.
    "filters": ["z", "i"],
    # Need at least this many input frames to build (else warn / blanks_fallback).
    # Absolute floor in code is 2. With 3 frames the per-pixel noise of the
    # master is ~σ_sky/√3 per frame — typically several × the fringe
    # amplitude — so more frames help far more than any tuning below.
    "min_frames": 3,

    # --- Frame selection mode ---
    # blanks  = only OBJECT matching blank_substrings (best when deep + dithered)
    # science = non-blank science only (sharper PSFs; need many fields)
    # all     = blanks + science, then similarity cuts below
    "select": "all",
    # Substrings matched against OBJECT (case-insensitive) to flag blank/sky frames.
    "blank_substrings": ["blank", "sky", "empty"],
    # If select=blanks but n < min_frames: "all" | "science" | "none"
    "blanks_fallback": "all",

    # --- Similarity cuts (set 0 / null to disable that cut) ---
    # Keep frames within ± this of the median AIRMASS. Tighter = more uniform
    # airglow; too tight = too few frames.
    "airmass_tol": 0.15,
    # Keep frames within ± this many hours of median DATE-OBS / MJD-OBS.
    # Fringe amplitude drifts through the night; ~1–2 h is typical.
    "time_tol_hours": 1.5,
    # Keep frames with |sky - median_sky| / median_sky ≤ this. Stops mixing
    # short/shallow exposures with deep blanks after /sky scaling. 0 = off.
    "sky_tol_frac": 0.3,
    # Drop frames with sky_median below this (ADU). Shallow → poor fringe S/N.
    # 0 = off.
    "min_sky": 40.0,
    # Drop frames with EXPTIME below this (seconds). 0 = off.
    "min_exptime": 20.0,

    # --- Per-frame star mask (before combine) ---
    # Core detection in σ above clipped sky. KEEP HIGH (≥4–5) so fringe peaks
    # are not masked. Lower → more mask / risk eating fringe; higher → ghosts.
    "star_nsigma": 5.0,
    # Grow mask into connected pixels above this σ (soft wings), then apply
    # hard circular dilate of dilate_pix.
    "wing_nsigma": 2.5,
    # Hard dilate radius in pixels around cores/wings.
    #   ↑ larger → fewer ghosts, but more pixels with 0 samples (see hole_fill)
    #   ↓ smaller → more pixels keep real data; soft PSF wings may leak
    # Soft blanks: 12–20 is typical, not 30+.
    "dilate_pix": 16,
    # If >0, detect on Gaussian-smoothed copy (σ px). Helps ultra-soft blobs
    # but can fold fringe into the mask — leave 0 unless needed.
    "star_smooth_sigma": 0.0,

    # --- Combine ---
    # Per pixel: drop this many brightest *unmasked* samples, then median.
    # 0 = plain masked median (DEFAULT, unbiased).
    # WARNING — rejection is BIASED with few frames. Taking the min of 3
    # noisy samples sits ~0.85σ below the true sky; min of 2 only ~0.56σ;
    # a lone sample 0σ. The bias therefore depends on how many frames were
    # masked at that pixel, so the master comes out BRIGHTER around every
    # masked star (measured +1.5…+3.5 % of sky on 2016OCT11 z with
    # reject_high=2) — this looked like "star halos" but was pure noise
    # statistics. Only use ≥1 when n_frames ≥ ~8 so the bias is small and
    # nearly uniform.
    "reject_high": 0,

    # --- Pixels with NO unmasked sample in any frame ---
    # Nothing is interpolated or smoothed (fake smooth islands are not data).
    #   data = re-measure using a *loose* mask (star cores only, no wing
    #          growth / dilation) → median of those real samples; whatever is
    #          still empty becomes 0 (= no correction there).
    #   zero = set straight to 0.
    # The per-pixel sample count is written to the NSAMP extension.
    "hole_fill": "data",

    # --- Post-combine residual-ghost cleanup (OFF by default) ---
    # Subtract a large-scale Gaussian from the stack; flag compact *positive*
    # leftovers (ghosts), dilate, and set them to 0 (no correction). Fringe
    # (broad) stays in the smooth component and is not flagged.
    # σ cut on (stack − smooth). 0 = off. Lower = more aggressive.
    "residual_nsigma": 0.0,
    # Gaussian σ (px) for the large-scale model. ≫ ghost size, ≲ fringe scale.
    "residual_smooth_sigma": 15.0,
    # Extra dilate (px) on the residual-ghost mask.
    "residual_dilate": 8,
}

SELECT_MODES = ("blanks", "science", "all")


def fringe_config(cfg: dict) -> dict[str, Any]:
    out = dict(DEFAULT_FRINGE)
    user = dict(cfg.get("fringe") or {})
    out.update(user)

    # Legacy prefer_blank / include_all_if_few_blank → select / blanks_fallback
    if "select" not in user:
        if user.get("prefer_blank") is False:
            out["select"] = "all"
        else:
            out["select"] = "blanks"
    if "blanks_fallback" not in user and "include_all_if_few_blank" in user:
        out["blanks_fallback"] = "all" if user.get("include_all_if_few_blank") else "none"

    mode = str(out.get("select", "blanks")).lower().strip()
    if mode not in SELECT_MODES:
        raise ValueError(
            f"fringe.select must be one of {SELECT_MODES}, got {mode!r}"
        )
    out["select"] = mode
    fb = str(out.get("blanks_fallback", "all")).lower().strip()
    if fb not in ("all", "science", "none"):
        raise ValueError(
            f"fringe.blanks_fallback must be all|science|none, got {fb!r}"
        )
    out["blanks_fallback"] = fb
    return out


def _is_blank_object(obj: str, blank_substrings: list[str]) -> bool:
    text = (obj or "").lower()
    return any(s in text for s in blank_substrings)


def _frame_airmass(header) -> float | None:
    for key in ("AIRMASS", "SECZ", "AIRMASS1"):
        if key in header and header[key] not in (None, ""):
            try:
                return float(header[key])
            except (TypeError, ValueError):
                pass
    return None


def _ut_hours(header) -> float | None:
    """Hours since midnight from ``UT`` / ``UTC`` / ``TIME-OBS``.

    Hall writes ``UT`` as a float hour (7.345647). HDI-style headers use
    ``HH:MM:SS`` in ``TIME-OBS`` or a ``DATE-OBS`` that already has a time.
    """
    for key in ("UT", "UTC", "TIME-OBS", "UT-TIME"):
        raw = header.get(key)
        if raw in (None, ""):
            continue
        if isinstance(raw, (int, float)):
            return float(raw)
        text = str(raw).strip().replace("T", " ").split()[-1]
        parts = text.split(":")
        try:
            if len(parts) >= 2:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2]) if len(parts) > 2 else 0.0
                return hours + minutes / 60.0 + seconds / 3600.0
            return float(text)
        except (TypeError, ValueError):
            continue
    return None


def _format_hms(hours: float) -> str:
    if not np.isfinite(hours):
        return "—"
    total = int(round(float(hours) * 3600.0)) % (24 * 3600)
    hh, rem = divmod(total, 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _frame_mjd(header) -> float | None:
    if header.get("MJD-OBS") not in (None, ""):
        try:
            return float(header["MJD-OBS"])
        except (TypeError, ValueError):
            pass
    raw = header.get("DATE-OBS")
    ut_h = _ut_hours(header)
    if not raw:
        return None
    text = str(raw).strip()
    try:
        from astropy.time import Time

        if "T" in text or (" " in text and ":" in text):
            return float(Time(text).mjd)
        t0 = float(Time(text).mjd)
        if ut_h is not None:
            return t0 + float(ut_h) / 24.0
        return t0
    except Exception:
        try:
            from astropy.time import Time

            return float(Time(text).mjd)
        except Exception:
            return None


def _frame_time_str(header, mjd: float | None) -> str:
    raw = str(header.get("DATE-OBS") or "").strip()
    if "T" in raw:
        return raw.replace("T", " ")[:19]
    ut_h = _ut_hours(header)
    if ut_h is not None:
        return _format_hms(ut_h)
    if mjd is not None and np.isfinite(mjd):
        try:
            from astropy.time import Time

            return str(Time(mjd, format="mjd").isot).replace("T", " ")[:19]
        except Exception:
            pass
    return raw[:19] if raw else "—"


def _frame_pointing_str(header) -> tuple[str, str]:
    pointing = parse_pointing(header)
    if pointing is not None:
        import astropy.units as u

        ra = pointing.ra.to_string(unit=u.hourangle, sep=":", precision=1, pad=True)
        dec = pointing.dec.to_string(
            unit=u.deg, sep=":", precision=0, pad=True, alwayssign=True
        )
        return str(ra), str(dec)
    ra = header.get("RA") or header.get("TELRA") or header.get("RASTRNG")
    dec = header.get("DEC") or header.get("TELDEC") or header.get("DECSTRNG")
    ra_s = str(ra).strip() if ra not in (None, "") else "—"
    dec_s = str(dec).strip() if dec not in (None, "") else "—"
    return ra_s or "—", dec_s or "—"


def _split_blank_science(
    paths: list[Path],
    blank_substrings: list[str],
) -> tuple[list[Path], list[Path]]:
    blanks: list[Path] = []
    science: list[Path] = []
    for path in sorted(paths):
        obj = parse_science_filename(path)["object"]
        if _is_blank_object(obj, blank_substrings):
            blanks.append(path)
        else:
            science.append(path)
    return blanks, science


def collect_fringe_catalog(
    paths: list[Path],
    *,
    blank_substrings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """One row per frame: time, pointing, airmass, sky — for cuts and the picker."""
    blanks = [s.lower() for s in (blank_substrings or ["blank", "sky", "empty"])]
    rows: list[dict[str, Any]] = []
    for path in paths:
        parsed = parse_science_filename(path)
        obj = parsed.get("object") or ""
        seq = parsed.get("seq") or ""
        try:
            hdr = fits.getheader(path, 0)
        except Exception:
            hdr = fits.Header()
        if not obj:
            obj = str(hdr.get("OBJECT") or "")
        label = f"{obj}.{seq}" if seq else obj
        exp = None
        if hdr.get("EXPTIME") not in (None, ""):
            try:
                exp = float(hdr["EXPTIME"])
            except (TypeError, ValueError):
                exp = None
        sky = float("nan")
        try:
            data = np.asarray(fits.getdata(path), dtype=float)
            finite = data[np.isfinite(data)]
            if finite.size:
                sky = float(np.median(finite[:: max(1, finite.size // 200_000)]))
        except Exception:
            sky = float("nan")
        mjd = _frame_mjd(hdr)
        ra, dec = _frame_pointing_str(hdr)
        rows.append(
            {
                "path": path,
                "file": path.name,
                "object": label or obj or path.stem,
                "blank": _is_blank_object(obj, blanks),
                "airmass": _frame_airmass(hdr),
                "mjd": mjd,
                "time": _frame_time_str(hdr, mjd),
                "ra": ra,
                "dec": dec,
                "exptime": exp,
                "sky": sky,
            }
        )
    return rows


def _fmt_num(value, *, width: int, spec: str, missing: str = "—") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return f"{missing:>{width}}"
    return format(value, spec)


def format_fringe_catalog(
    rows: list[dict[str, Any]],
    kept: list[Path] | None = None,
) -> str:
    """Numbered table: UT, RA, Dec, airmass, exposure, sky. ``*`` = auto-kept."""
    if not rows:
        return "  (no candidate frames)"
    kept_set = {Path(p) for p in (kept or [])}
    obj_w = min(22, max(6, *(len(str(r["object"])) for r in rows)))
    file_w = min(36, max(4, *(len(r["file"]) for r in rows)))
    head = (
        f"  {'#':>3}  {'*':>1}  {'UT':<8}  {'RA':<11}  {'Dec':<10}  "
        f"{'AM':>5}  {'t(s)':>6}  {'sky':>7}  {'object':<{obj_w}}  file"
    )
    lines = [head]
    for i, row in enumerate(rows, start=1):
        mark = "*" if Path(row["path"]) in kept_set else " "
        am = _fmt_num(row.get("airmass"), width=5, spec=">5.2f")
        exp = _fmt_num(row.get("exptime"), width=6, spec=">6.1f")
        sky = _fmt_num(row.get("sky"), width=7, spec=">7.0f")
        obj = str(row.get("object") or "")[:obj_w]
        fname = row["file"] if len(row["file"]) <= file_w else row["file"][: file_w - 1] + "…"
        lines.append(
            f"  {i:3d}  {mark:>1}  {str(row.get('time') or '—'):<8}  "
            f"{str(row.get('ra') or '—'):<11}  {str(row.get('dec') or '—'):<10}  "
            f"{am}  {exp}  {sky}  {obj:<{obj_w}}  {fname}"
        )
    return "\n".join(lines)


def _cut_summary(
    *,
    airmass_tol: float,
    time_tol_hours: float,
    sky_tol_frac: float,
    min_sky: float,
    min_exptime: float,
) -> str:
    bits = []
    if airmass_tol > 0:
        bits.append(f"airmass ±{airmass_tol:g}")
    if time_tol_hours > 0:
        bits.append(f"time ±{time_tol_hours:g} h")
    if sky_tol_frac > 0:
        bits.append(f"sky ±{100 * sky_tol_frac:.0f}%")
    if min_sky > 0:
        bits.append(f"sky ≥{min_sky:g} ADU")
    if min_exptime > 0:
        bits.append(f"exptime ≥{min_exptime:g}s")
    return ", ".join(bits) if bits else "no similarity cuts"


def apply_similarity_cuts(
    rows: list[dict[str, Any]],
    *,
    airmass_tol: float | None = 0.3,
    time_tol_hours: float | None = 4.0,
    sky_tol_frac: float | None = 0.0,
    min_sky: float | None = 0.0,
    min_exptime: float | None = 0.0,
) -> list[Path]:
    """Apply airmass / time / sky / exposure cuts to a fringe catalog."""
    if not rows:
        return []
    kept = [r["path"] for r in rows]
    by_path = {r["path"]: r for r in rows}
    am_tol = float(airmass_tol) if airmass_tol not in (None, "") else 0.0
    t_tol_h = float(time_tol_hours) if time_tol_hours not in (None, "") else 0.0
    sky_frac = float(sky_tol_frac) if sky_tol_frac not in (None, "") else 0.0
    sky_min = float(min_sky) if min_sky not in (None, "") else 0.0
    exp_min = float(min_exptime) if min_exptime not in (None, "") else 0.0

    def _still(path: Path) -> dict[str, Any]:
        return by_path[path]

    if exp_min > 0:
        before = len(kept)
        kept = [
            p
            for p in kept
            if (exp := _still(p).get("exptime")) is not None and exp >= exp_min
        ]
        info(f"  exptime cut: ≥{exp_min:g}s  kept {len(kept)}/{before}")

    if sky_min > 0:
        before = len(kept)
        kept = [
            p
            for p in kept
            if np.isfinite(_still(p).get("sky")) and _still(p)["sky"] >= sky_min
        ]
        info(f"  min_sky cut: ≥{sky_min:g} ADU  kept {len(kept)}/{before}")

    if am_tol > 0:
        ams = [
            r["airmass"]
            for r in (_still(p) for p in kept)
            if r.get("airmass") is not None and np.isfinite(r["airmass"])
        ]
        if ams:
            am0 = float(np.median(ams))
            before = len(kept)
            kept = [
                p
                for p in kept
                if (a := _still(p).get("airmass")) is not None and abs(a - am0) <= am_tol
            ]
            info(
                f"  airmass cut: median={am0:.3f} ±{am_tol:.3f}  "
                f"kept {len(kept)}/{before}"
            )

    if t_tol_h > 0 and kept:
        mjds = [
            r["mjd"]
            for r in (_still(p) for p in kept)
            if r.get("mjd") is not None and np.isfinite(r["mjd"])
        ]
        if mjds:
            m0 = float(np.median(mjds))
            dt = t_tol_h / 24.0
            before = len(kept)
            kept = [
                p
                for p in kept
                if (m := _still(p).get("mjd")) is not None and abs(m - m0) <= dt
            ]
            info(
                f"  time cut: ±{t_tol_h:.2g} h around median DATE-OBS  "
                f"kept {len(kept)}/{before}"
            )

    if sky_frac > 0 and kept:
        skies = [
            r["sky"]
            for r in (_still(p) for p in kept)
            if np.isfinite(r.get("sky"))
        ]
        if skies:
            s0 = float(np.median(skies))
            before = len(kept)
            if s0 > 0:
                kept = [
                    p
                    for p in kept
                    if np.isfinite(_still(p).get("sky"))
                    and abs(_still(p)["sky"] - s0) / s0 <= sky_frac
                ]
            info(
                f"  sky cut: median={s0:.4g} ±{100 * sky_frac:.0f}%  "
                f"kept {len(kept)}/{before}"
            )

    return kept


def filter_similar_conditions(
    paths: list[Path],
    *,
    airmass_tol: float | None = 0.3,
    time_tol_hours: float | None = 4.0,
    sky_tol_frac: float | None = 0.0,
    min_sky: float | None = 0.0,
    min_exptime: float | None = 0.0,
    blank_substrings: list[str] | None = None,
) -> list[Path]:
    """Keep frames with similar observing / sky conditions.

    Fringe amplitude and shape track airglow. Mixing very different airmass,
    time of night, or sky level (short vs deep exposures) blurs the master.
    Tolerances ≤0 disable that cut.

    ``sky_tol_frac`` uses a fast median sky estimate from the science pixels
    (same idea as the scale used when stacking). No moon keywords on HDI.
    """
    rows = collect_fringe_catalog(paths, blank_substrings=blank_substrings)
    return apply_similarity_cuts(
        rows,
        airmass_tol=airmass_tol,
        time_tol_hours=time_tol_hours,
        sky_tol_frac=sky_tol_frac,
        min_sky=min_sky,
        min_exptime=min_exptime,
    )


def _ask_cut_value(label: str, current: float) -> float:
    from .prompt import ask

    raw = ask(label, f"{current:g}")
    try:
        return float(raw)
    except ValueError:
        warn(f"  not a number ({raw!r}); keeping {current:g}")
        return current


def offer_fringe_frames(
    rows: list[dict[str, Any]],
    kept: list[Path],
    *,
    cuts: dict[str, Any],
    min_frames: int = 2,
    interactive: bool = True,
    label: str = "",
) -> list[Path]:
    """Print the candidate list; on a TTY, let the user pick or retune cuts."""
    if not rows:
        return []

    cuts = dict(cuts)
    current = list(kept)
    filt = f"filter '{label}' — " if label else ""

    while True:
        print()
        print(format_fringe_catalog(rows, current))
        print(
            dim(
                "  * kept by "
                + _cut_summary(
                    airmass_tol=float(cuts.get("airmass_tol") or 0),
                    time_tol_hours=float(cuts.get("time_tol_hours") or 0),
                    sky_tol_frac=float(cuts.get("sky_tol_frac") or 0),
                    min_sky=float(cuts.get("min_sky") or 0),
                    min_exptime=float(cuts.get("min_exptime") or 0),
                )
            )
        )
        if not interactive:
            return current

        from .prompt import ask_index_picks, choose

        choice = choose(
            f"Fringe frames ({filt}{len(current)}/{len(rows)} would be stacked)",
            [
                ("auto", f"Use the {len(current)} auto-cut frame(s)"),
                ("pick", "Pick from the list (time and coordinates above)"),
                ("all", f"Use all {len(rows)} listed frames"),
                ("cuts", "Change airmass / time / sky cuts and re-list"),
            ],
            default="auto",
        )
        if choice == "auto":
            return current
        if choice == "all":
            return [r["path"] for r in rows]
        if choice == "cuts":
            cuts["airmass_tol"] = _ask_cut_value(
                "airmass_tol (0 = off)", float(cuts.get("airmass_tol") or 0)
            )
            cuts["time_tol_hours"] = _ask_cut_value(
                "time_tol_hours (0 = off)", float(cuts.get("time_tol_hours") or 0)
            )
            cuts["sky_tol_frac"] = _ask_cut_value(
                "sky_tol_frac (0 = off)", float(cuts.get("sky_tol_frac") or 0)
            )
            cuts["min_sky"] = _ask_cut_value(
                "min_sky ADU (0 = off)", float(cuts.get("min_sky") or 0)
            )
            cuts["min_exptime"] = _ask_cut_value(
                "min_exptime s (0 = off)", float(cuts.get("min_exptime") or 0)
            )
            current = apply_similarity_cuts(rows, **cuts)
            continue

        default_idx = [
            i for i, row in enumerate(rows, start=1) if row["path"] in set(current)
        ]
        picked = ask_index_picks(
            len(rows),
            prompt="Frame numbers (e.g. 1 3 5-8 or all)",
            default=default_idx or list(range(1, len(rows) + 1)),
        )
        chosen = [rows[i - 1]["path"] for i in picked]
        if len(chosen) < min(2, min_frames):
            warn(f"  need ≥2 frames, got {len(chosen)}")
            continue
        return chosen


def select_fringe_inputs(
    paths: list[Path],
    *,
    select: str = "blanks",
    blank_substrings: list[str] | None = None,
    blanks_fallback: str = "all",
    min_frames: int = 3,
    airmass_tol: float | None = 0.3,
    time_tol_hours: float | None = 4.0,
    sky_tol_frac: float | None = 0.0,
    min_sky: float | None = 0.0,
    min_exptime: float | None = 0.0,
    include_all_if_few_blank: bool | None = None,
    interactive: bool | None = None,
    label: str = "",
) -> list[Path]:
    """Choose frames for the fringe master.

    ``select``:
      * ``blanks``  — blank/sky OBJECT only (optional fallback if too few)
      * ``science`` — non-blank science only
      * ``all``     — everything, then similarity cuts

    When ``interactive`` (default: stdin is a TTY) a table of UT / RA / Dec
    is printed and you can accept the auto cuts, pick rows, or retune the cuts.
    """
    blank_substrings = [
        s.lower() for s in (blank_substrings or ["blank", "sky", "empty"])
    ]
    mode = str(select).lower().strip()
    if include_all_if_few_blank is False:
        blanks_fallback = "none"
    elif include_all_if_few_blank is True and blanks_fallback == "none":
        blanks_fallback = "all"
    if interactive is None:
        interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())

    blanks, science = _split_blank_science(paths, blank_substrings)
    info(
        f"  available: {len(blanks)} blank/sky, {len(science)} other science  "
        f"(select={mode})"
    )
    sim_kw = dict(
        airmass_tol=airmass_tol,
        time_tol_hours=time_tol_hours,
        sky_tol_frac=sky_tol_frac,
        min_sky=min_sky,
        min_exptime=min_exptime,
    )

    pool: list[Path]
    if mode == "blanks":
        if len(blanks) >= min_frames:
            pool = list(blanks)
            info(f"  pool: {len(pool)} blank/sky")
        elif blanks_fallback == "none":
            warn(
                f"  only {len(blanks)} blank(s) (< min_frames={min_frames}); "
                f"no fallback (blanks_fallback=none)"
            )
            pool = list(blanks)
        elif blanks_fallback == "science":
            warn(
                f"  only {len(blanks)} blank(s) (<{min_frames}); "
                f"falling back to non-blank science"
            )
            pool = list(science)
        else:
            warn(
                f"  only {len(blanks)} blank(s) (<{min_frames}); "
                f"falling back to all frames with similarity cuts"
            )
            pool = list(blanks) + list(science)
    elif mode == "science":
        pool = list(science)
        info(f"  pool: {len(pool)} non-blank science (blanks avoided)")
    else:
        pool = list(blanks) + list(science)
        info(f"  pool: {len(pool)} frames (blanks + science)")

    rows = collect_fringe_catalog(pool, blank_substrings=blank_substrings)
    chosen = apply_similarity_cuts(rows, **sim_kw)
    chosen = offer_fringe_frames(
        rows,
        chosen,
        cuts=sim_kw,
        min_frames=min_frames,
        interactive=interactive,
        label=label,
    )
    info(f"  using {len(chosen)} frame(s) for the fringe master")
    return chosen


def _sky_median(data: np.ndarray) -> float:
    """Robust sky level via Astropy ``sigma_clipped_stats`` (3σ).

    Used both to normalize frames when building the fringe master and as the
    scale factor when applying ``science − sky × F`` (IRAF-style).
    Docs: https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html
    """
    _, med, _ = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    return float(med)


def star_mask(
    data: np.ndarray,
    *,
    nsigma: float = 5.0,
    dilate_pix: int = 12,
    smooth_sigma: float = 0.0,
    wing_nsigma: float = 2.5,
) -> np.ndarray:
    """Boolean mask: True where stars / bright pixels should be ignored.

    Keep dilation **modest**. Large hard dilates leave pixels with no sample in
    any frame; those get no fringe value (see ``hole_fill``), not an invented one.
    ``dilate_pix=0`` gives the *loose* cores-only mask used for hole fallback.
    """
    work = np.asarray(data, dtype=float)
    if smooth_sigma and float(smooth_sigma) > 0:
        from scipy.ndimage import gaussian_filter

        work = gaussian_filter(work, sigma=float(smooth_sigma))

    _, med, std = sigma_clipped_stats(work, sigma=3.0, maxiters=5)
    std = float(std) if np.isfinite(std) and std > 0 else float(mad_std(work)) or 1.0
    finite = np.isfinite(work)
    cores = finite & (work > med + float(nsigma) * std)
    bright = cores.copy()

    max_r = int(max(0, dilate_pix))
    if max_r > 0 and cores.any() and float(wing_nsigma) > 0:
        wing_allowed = finite & (work > med + float(wing_nsigma) * std)
        footprint = np.ones((3, 3), dtype=bool)
        for _ in range(min(max_r, 32)):
            grown = binary_dilation(bright, structure=footprint) & wing_allowed
            if np.array_equal(grown, bright):
                break
            bright = grown

    if max_r > 0 and bright.any():
        yy, xx = np.ogrid[-max_r : max_r + 1, -max_r : max_r + 1]
        disk = xx * xx + yy * yy <= max_r * max_r
        bright = binary_dilation(bright, structure=disk)

    return bright | ~np.isfinite(data)


def _masked_reject_combine(
    arr: np.ndarray,
    msk: np.ndarray,
    *,
    reject_high: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel combine: drop highest ``reject_high`` unmasked values, then median.

    Returns ``(combined, n_samples)`` where ``n_samples`` is the number of
    unmasked inputs per pixel *before* rejection. Pixels with no sample are NaN.

    ESO fringe maps average with rejection of the highest values, but they use
    many frames. With 3–4 frames the order-statistic bias of the rejection
    depends on how many frames were masked at each pixel and prints bright
    halos around every star — keep ``reject_high=0`` unless n_frames ≳ 8.
    """
    import warnings

    rej = int(max(0, reject_high))
    work = np.where(msk, np.nan, np.asarray(arr, dtype=float))
    n_samples = np.sum(np.isfinite(work), axis=0).astype(np.int16)

    for _ in range(rej):
        n_fin = np.sum(np.isfinite(work), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                mx = np.nanmax(work, axis=0)
        can = n_fin > 1
        if not np.any(can):
            break
        is_max = can[np.newaxis, ...] & np.isfinite(work) & (work == mx[np.newaxis, ...])
        seen = np.cumsum(is_max, axis=0)
        drop = is_max & (seen == 1)
        work = np.where(drop, np.nan, work)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            return np.nanmedian(work, axis=0), n_samples


def _fill_holes_from_data(
    combined: np.ndarray,
    arr: np.ndarray,
    loose_msk: np.ndarray,
    *,
    clip_nsigma: float = 5.0,
) -> tuple[np.ndarray, int, int]:
    """Fill all-masked pixels with *real* samples only — never interpolate.

    Where the strict mask left no sample, take the median of the inputs that
    survive the loose (cores-only) mask, but only if that value is within
    ``clip_nsigma`` robust σ of the stack's sky distribution (rejects saturated
    / negative / dead pixels inside star cores). Returns ``(filled,
    n_from_loose, n_still_empty)``; still-empty pixels stay NaN (caller → 0).
    """
    import warnings

    out = np.asarray(combined, dtype=float).copy()
    bad = ~np.isfinite(out)
    if not bad.any():
        return out, 0, 0
    good = out[~bad]
    med = float(np.median(good))
    sig = float(mad_std(good)) or 1e-6
    loose = np.where(loose_msk, np.nan, np.asarray(arr, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            loose_med = np.nanmedian(loose[:, bad], axis=0)
            plausible = np.abs(loose_med - med) <= float(clip_nsigma) * sig
    loose_med = np.where(plausible, loose_med, np.nan)
    out[bad] = loose_med
    n_loose = int(np.isfinite(loose_med).sum())
    n_empty = int(bad.sum()) - n_loose
    return out, n_loose, n_empty


def _clean_residual_stars(
    combined: np.ndarray,
    *,
    nsigma: float = 4.0,
    smooth_sigma: float = 15.0,
    dilate: int = 8,
) -> tuple[np.ndarray, float]:
    """Flag compact positive leftovers after the stack (soft star ghosts).

    Subtract a large-scale Gaussian so real fringe (broad) is not flagged; mask
    only compact positive bumps and set them to NaN (caller zeroes them = no
    correction). Nothing is interpolated. Returns (cleaned, frac_masked).
    """
    from scipy.ndimage import gaussian_filter

    work = np.asarray(combined, dtype=float)
    if not np.isfinite(work).any() or not nsigma or float(nsigma) <= 0:
        return work, 0.0

    # Replace NaNs temporarily for filtering
    med0 = float(np.nanmedian(work))
    seeded = np.where(np.isfinite(work), work, med0)
    large = gaussian_filter(seeded, sigma=float(smooth_sigma))
    resid = seeded - large
    _, rmed, rstd = sigma_clipped_stats(resid, sigma=3.0, maxiters=5)
    rstd = float(rstd) if np.isfinite(rstd) and rstd > 0 else float(mad_std(resid)) or 1e-6
    hot = resid > rmed + float(nsigma) * rstd
    if dilate and hot.any():
        r = int(max(0, dilate))
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        disk = xx * xx + yy * yy <= r * r
        hot = binary_dilation(hot, structure=disk)
    frac = float(hot.mean())
    if not hot.any():
        return work, 0.0
    out = work.copy()
    out[hot] = np.nan
    return out, frac


def build_master_fringe(
    paths: list[Path],
    *,
    star_nsigma: float = 5.0,
    dilate_pix: int = 16,
    star_smooth_sigma: float = 0.0,
    wing_nsigma: float = 2.5,
    reject_high: int = 0,
    hole_fill: str = "data",
    residual_nsigma: float = 0.0,
    residual_smooth_sigma: float = 15.0,
    residual_dilate: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sky-scaled, star-masked median stack → zero-mean ``F``.

    Methodology (README §6 / IRAF mkfringecor)
    ------------------------------------------
    1. ``scaled = frame / sky_median`` — sky ≈ 1 for every frame.
    2. Mask star cores (+ wings, modest dilate) per frame.
    3. Per pixel: median of the unmasked samples (optional high rejection —
       biased with few frames, off by default).
    4. Pixels with no sample: re-measure with a cores-only mask, else 0.
       **No interpolation or smoothing anywhere** — every non-zero pixel of
       the master is a measured sky value.
    5. ``F = S − median(S)`` — zero-mean fraction-of-sky template.

    ``meta['n_samples']`` carries the per-pixel sample-count map (NSAMP ext).
    """
    if len(paths) < 2:
        raise RuntimeError("Need at least 2 frames to build a fringe master")

    stack = []
    masks = []
    loose_masks = []
    sky_levels = []
    for path in paths:
        step(path.name)
        data = np.asarray(fits.getdata(path), dtype=float)
        sky = _sky_median(data)
        if not np.isfinite(sky) or abs(sky) < 1e-6:
            warn(f"    skip — bad sky median {sky}")
            continue
        scaled = data / sky
        mask = star_mask(
            data,
            nsigma=star_nsigma,
            dilate_pix=dilate_pix,
            smooth_sigma=star_smooth_sigma,
            wing_nsigma=wing_nsigma,
        )
        frac_masked = float(mask.mean())
        note = ""
        if frac_masked > 0.35:
            note = "  ⚠ high — many pixels will have no sample; lower dilate_pix"
        print(dim(f"    sky_median={sky:.4g}  masked={100*frac_masked:.1f}%{note}"))
        stack.append(scaled)
        masks.append(mask)
        # Loose mask (cores only, no growth/dilate) for the hole fallback.
        loose_masks.append(
            star_mask(data, nsigma=star_nsigma, dilate_pix=0, smooth_sigma=star_smooth_sigma)
        )
        sky_levels.append(sky)

    if len(stack) < 2:
        raise RuntimeError("Fewer than 2 usable frames for fringe stack")

    n_frames = len(stack)
    arr = np.stack(stack, axis=0)
    msk = np.stack(masks, axis=0)
    if int(reject_high) > 0 and n_frames < 8:
        warn(
            f"    reject_high={int(reject_high)} with only {n_frames} frames biases the "
            "master bright around masked stars (order statistics); 0 is recommended"
        )
    combined, n_samples = _masked_reject_combine(arr, msk, reject_high=int(reject_high))
    npix = combined.size
    for n in range(n_frames + 1):
        frac = float(np.mean(n_samples == n))
        if frac > 0:
            print(dim(f"    pixels with {n}/{n_frames} unmasked samples: {100*frac:6.2f}%"))

    hole_mode = str(hole_fill or "data").lower().strip()
    n_loose = n_empty = 0
    if hole_mode == "data":
        combined, n_loose, n_empty = _fill_holes_from_data(
            combined, arr, np.stack(loose_masks, axis=0)
        )
    else:
        n_empty = int((~np.isfinite(combined)).sum())

    combined, resid_frac = _clean_residual_stars(
        combined,
        nsigma=float(residual_nsigma),
        smooth_sigma=float(residual_smooth_sigma),
        dilate=int(residual_dilate),
    )
    if resid_frac > 0:
        print(dim(f"    residual-ghost cleanup flagged {100*resid_frac:.2f}% → 0 (no correction)"))

    # Zero-mean template. Pixels with no measurement get exactly 0 = "do not
    # correct here" — never an interpolated or smoothed guess.
    fringe = combined - float(np.nanmedian(combined))
    empty = ~np.isfinite(fringe)
    fringe[empty] = 0.0
    if n_loose or n_empty:
        print(
            dim(
                f"    no-sample pixels: {100*n_loose/npix:.2f}% re-measured with cores-only mask, "
                f"{100*n_empty/npix:.2f}% left at 0"
            )
        )

    meta = {
        "n_frames": n_frames,
        "sky_levels": sky_levels,
        "input_files": [p.name for p in paths[:n_frames]],
        "star_nsigma": star_nsigma,
        "wing_nsigma": wing_nsigma,
        "dilate_pix": dilate_pix,
        "star_smooth_sigma": star_smooth_sigma,
        "reject_high": int(reject_high),
        "hole_fill": hole_mode,
        "n_samples": n_samples,
        "frac_loose_fill": n_loose / npix,
        "frac_empty": n_empty / npix,
        "residual_frac": resid_frac,
        "fringe_rms": float(np.nanstd(fringe)),
        "fringe_ptp": float(np.ptp(fringe)),
        "ref_noirlab": FRINGE_REFS["noirlab_soi"],
        "ref_iraf": FRINGE_REFS["iraf_mkfringecor"],
    }
    return fringe.astype(np.float32), meta


def fringe_path(cfg: dict, filter_name: str) -> Path:
    return Path(cfg["paths"]["masters_dir"]) / f"master_fringe_{filter_name}.fits"


def save_fringe_preview(
    fringe: np.ndarray,
    out_path: Path,
    *,
    title: str,
    n_samples: np.ndarray | None = None,
) -> Path:
    """Preview PNG: master (left) and, if given, unmasked-samples map (right)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, max(fringe.shape) // 1024)
    plot = fringe[::step, ::step]
    lim = float(np.nanpercentile(np.abs(plot), 99)) or 1e-3
    ncol = 2 if n_samples is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 5.5), squeeze=False)
    ax = axes[0, 0]
    im = ax.imshow(plot, origin="lower", cmap="gray", vmin=-lim, vmax=lim)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of sky")
    if n_samples is not None:
        ax2 = axes[0, 1]
        ns = np.asarray(n_samples)[::step, ::step]
        im2 = ax2.imshow(ns, origin="lower", cmap="viridis", vmin=0, vmax=int(ns.max()))
        ax2.set_title("unmasked samples per pixel (0 = no data → F=0)")
        fig.colorbar(im2, ax=ax2, fraction=0.046, label="frames")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def make_master_fringe(
    cfg: dict,
    *,
    filters: list[str] | None = None,
    interactive: bool | None = None,
) -> dict[str, Path]:
    """Build ``master_fringe_<filter>.fits`` under masters_dir.

    Must run **after** science reduce (bias+flat). Fringe is additive and
    assumes flat-corrected ADU. See README Methodology §6.
    """
    fcfg = fringe_config(cfg)
    want = filters or list(fcfg.get("filters") or ["z"])
    fringe_flags = ["--select", str(fcfg.get("select", "all"))]
    for filt_name in want:
        fringe_flags.extend(["--filter", str(filt_name)])
    show_ccd(cfg, "fringe", *fringe_flags)
    paths = iter_science_fits(cfg)
    by_filt = group_science_by_filter(paths)
    masters_dir = Path(cfg["paths"]["masters_dir"])
    masters_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = masters_dir / "diagnostics"
    outputs: dict[str, Path] = {}

    for filt in want:
        subheading(f"Master fringe — filter '{filt}'")
        flist = by_filt.get(filt) or []
        if not flist:
            warn(f"  no science frames for filter '{filt}'")
            continue
        chosen = select_fringe_inputs(
            flist,
            select=str(fcfg.get("select", "all")),
            blank_substrings=list(fcfg.get("blank_substrings") or []),
            blanks_fallback=str(fcfg.get("blanks_fallback", "all")),
            min_frames=int(fcfg.get("min_frames", 3)),
            airmass_tol=fcfg.get("airmass_tol", 0.15),
            time_tol_hours=fcfg.get("time_tol_hours", 1.5),
            sky_tol_frac=fcfg.get("sky_tol_frac", 0.3),
            min_sky=fcfg.get("min_sky", 40.0),
            min_exptime=fcfg.get("min_exptime", 20.0),
            interactive=interactive,
            label=str(filt),
        )
        if len(chosen) < 2:
            warn(f"  need ≥2 frames, found {len(chosen)}")
            continue

        fringe, meta = build_master_fringe(
            chosen,
            star_nsigma=float(fcfg.get("star_nsigma", 5.0)),
            dilate_pix=int(fcfg.get("dilate_pix", 16)),
            star_smooth_sigma=float(fcfg.get("star_smooth_sigma", 0.0) or 0.0),
            wing_nsigma=float(fcfg.get("wing_nsigma", 2.5)),
            reject_high=int(fcfg.get("reject_high", 0) or 0),
            hole_fill=str(fcfg.get("hole_fill", "data")),
            residual_nsigma=float(fcfg.get("residual_nsigma", 0.0) or 0.0),
            residual_smooth_sigma=float(fcfg.get("residual_smooth_sigma", 15.0)),
            residual_dilate=int(fcfg.get("residual_dilate", 8)),
        )
        hdr = fits.Header()
        hdr["IMAGETYP"] = "FRINGE"
        hdr["FILTER"] = filt
        hdr["BUNIT"] = ("", "fraction of sky, zero-mean")
        hdr["NCOMBINE"] = (meta["n_frames"], "input frames")
        hdr["FRGRMS"] = (meta["fringe_rms"], "rms of master (fraction of sky)")
        hdr["FRGPTP"] = (meta["fringe_ptp"], "peak-to-peak of master")
        hdr["FRGNSIG"] = (float(meta["star_nsigma"]), "star core mask sigma")
        hdr["FRGWSIG"] = (float(meta["wing_nsigma"]), "wing growth sigma")
        hdr["FRGDIL"] = (int(meta["dilate_pix"]), "mask dilate radius px")
        hdr["FRGREJ"] = (int(meta["reject_high"]), "high samples rejected/pixel")
        hdr["FRGHOLE"] = (meta["hole_fill"], "no-sample pixel policy")
        hdr["FRGLOOSE"] = (round(meta["frac_loose_fill"], 6), "frac re-measured w/ cores-only mask")
        hdr["FRGEMPTY"] = (round(meta["frac_empty"], 6), "frac with no data (set 0)")
        hdr["FRGRESID"] = (round(meta["residual_frac"], 6), "frac zeroed by residual cleanup")
        for i, name in enumerate(meta["input_files"], 1):
            hdr[f"FRGIN{i:02d}"] = name
        hdr.add_history(
            "ccd_pipeline: fringe = median_over_frames(frame/sky, star-masked) - median; "
            f"reject_high={meta['reject_high']}; no interpolation/smoothing; "
            "no-sample pixels = 0. NSAMP ext = unmasked samples per pixel "
            f"(see {FRINGE_REFS['iraf_mkfringecor']})"
        )
        out = fringe_path(cfg, filt)
        nsamp = fits.ImageHDU(
            data=np.asarray(meta["n_samples"], dtype=np.int16), name="NSAMP"
        )
        nsamp.header["COMMENT"] = "unmasked input samples per pixel (0 = no data in master)"
        fits.HDUList([fits.PrimaryHDU(data=fringe, header=hdr), nsamp]).writeto(
            out, overwrite=True
        )
        outputs[filt] = out
        preview = save_fringe_preview(
            fringe,
            diag_dir / f"master_fringe_{filt}_diag.png",
            title=f"Master fringe '{filt}' (fraction of sky)",
            n_samples=meta["n_samples"],
        )
        success(
            f"\nWrote {out.name}  n={meta['n_frames']}  "
            f"rms={meta['fringe_rms']:.4g}  ptp={meta['fringe_ptp']:.4g}  "
            f"no-data={100*meta['frac_empty']:.2f}%"
        )
        info(f"  preview → {preview}")

    return outputs


def apply_fringe_to_frame(
    data: np.ndarray,
    fringe: np.ndarray,
    *,
    sky: float | None = None,
) -> tuple[np.ndarray, float]:
    """Subtract ``sky × fringe`` from a science frame (additive correction).

    Classic first-order scale (IRAF rmfringe / Astro-WISE)::

        science_corr = science − sky_median × F

    where ``F`` is the zero-mean fraction-of-sky master. Must be applied
    **after** flat-fielding. See FRINGE_REFS['iraf_rmfringe'].
    """
    if data.shape != fringe.shape:
        raise ValueError(f"Shape mismatch science {data.shape} vs fringe {fringe.shape}")
    if sky is None:
        sky = _sky_median(data)
    return data - float(sky) * fringe, float(sky)


def apply_master_fringe(
    cfg: dict,
    *,
    filters: list[str] | None = None,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Subtract scaled master fringe from calibrated science frames.

    Operates on already-reduced (bias+flat) science products. Sets ``FRINGCOR``,
    ``FRINGSCL`` (sky used as scale), and HISTORY, then adds ``f`` to the
    filename prefix (``p-`` → ``pf-``, ``pw-`` → ``pfw-``). See README
    Methodology §6.
    """
    fcfg = fringe_config(cfg)
    want = filters or list(fcfg.get("filters") or ["z"])
    apply_flags: list[str] = ["--apply-only"]
    if overwrite:
        apply_flags.append("--overwrite")
    if limit is not None:
        apply_flags.extend(["--limit", str(limit)])
    for filt_name in want:
        apply_flags.extend(["--filter", str(filt_name)])
    show_ccd(cfg, "fringe", *apply_flags)
    paths = iter_science_fits(cfg)
    by_filt = group_science_by_filter(paths)
    counts = {"ok": 0, "skip": 0, "fail": 0}

    for filt in want:
        subheading(f"Apply fringe — filter '{filt}'")
        fpath = fringe_path(cfg, filt)
        if not fpath.exists():
            warn(f"  missing {fpath.name} — run build first")
            counts["fail"] += 1
            continue
        fringe = np.asarray(fits.getdata(fpath), dtype=float)
        flist = list(by_filt.get(filt) or [])
        if limit is not None:
            flist = flist[:limit]
        print(f"  frames: {len(flist)}")
        for path in flist:
            outcome = "ok"
            sky = None
            with fits.open(path, mode="update", memmap=False) as hdul:
                hdr = hdul[0].header
                if hdr.get("FRINGCOR") and not overwrite:
                    outcome = "skip"
                else:
                    data = np.asarray(hdul[0].data, dtype=float)
                    # If re-applying, undo the previous sky×F using stored cards when
                    # the prior master still matches FRINGFIL (same path). Otherwise
                    # re-reduce science first — do not double-subtract.
                    if hdr.get("FRINGCOR") and overwrite:
                        prev_name = str(hdr.get("FRINGFIL") or "")
                        prev_scl = hdr.get("FRINGSCL")
                        prev_path = (
                            Path(cfg["paths"]["masters_dir"]) / prev_name
                            if prev_name
                            else None
                        )
                        if (
                            prev_path is not None
                            and prev_path.exists()
                            and prev_scl is not None
                            and prev_path.resolve() == fpath.resolve()
                        ):
                            data = data + float(prev_scl) * fringe
                        elif (
                            prev_path is not None
                            and prev_path.exists()
                            and prev_scl is not None
                        ):
                            try:
                                prev = np.asarray(fits.getdata(prev_path), dtype=float)
                                data = data + float(prev_scl) * prev
                            except Exception as exc:
                                warn(
                                    f"  FAIL {path.name}: cannot undo prior fringe ({exc}); "
                                    "re-reduce science then apply"
                                )
                                outcome = "fail"
                        else:
                            warn(
                                f"  FAIL {path.name}: FRINGCOR set but cannot undo "
                                f"(missing FRINGFIL/FRINGSCL); re-reduce science first"
                            )
                            outcome = "fail"
                    if outcome != "fail":
                        try:
                            # Additive: science − sky × F  (after flat; never before)
                            corrected, sky = apply_fringe_to_frame(data, fringe)
                        except Exception as exc:
                            warn(f"  FAIL {path.name}: {exc}")
                            outcome = "fail"
                        else:
                            hdul[0].data = corrected.astype(
                                hdul[0].data.dtype, copy=False
                            )
                            hdr["FRINGCOR"] = True
                            hdr["FRINGSCL"] = sky
                            hdr["FRINGFIL"] = fpath.name
                            if hasattr(hdr, "add_history"):
                                hdr.add_history(
                                    f"ccd_pipeline: subtract {sky:.4g} * {fpath.name} "
                                    f"(fringe); see {FRINGE_REFS['iraf_rmfringe']}"
                                )
                            hdul.flush()
            if outcome == "fail":
                counts["fail"] += 1
                continue
            dest = retitle_science_file(path, processed=True, fringe=True)
            remove_unprefixed_science_twin(dest)
            if outcome == "skip":
                print(dim(f"  SKIP {dest.name} (already fringe-corrected)"))
                counts["skip"] += 1
            else:
                print(dim(f"  OK   {dest.name}  sky_scale={sky:.4g}"))
                counts["ok"] += 1

    success(
        f"\nFringe apply: {counts['ok']} ok, {counts['skip']} skipped, {counts['fail']} failed"
    )
    return counts
