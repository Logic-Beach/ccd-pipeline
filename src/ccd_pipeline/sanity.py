"""Sanity checks and diagnostic plots for overscan + master bias / flats.

Professional practice: inspect calibrations **before** applying them to science
(README Methodology §4). Inspired by Astropy CCD Reduction Guide practice:

* Overscan column means into ``BIASSEC`` — flats reveal light leak; decide
  whether the header overscan region is trustworthy
  https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-08-Overscan.html
* Master bias / flat inspection — image, histograms, count statistics
  (bias median ≈ 0 after overscan; flats median ≈ 1 after inv-median combine)
  https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/02-04-Combine-bias-images-to-make-master.html
  https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-04-Combining-flats.html

Optional science-sky QC is delegated to ``science_qc`` (post-reduce flatness /
fringe indicators). Stats use Astropy ``sigma_clipped_stats`` /
``mad_std``: https://docs.astropy.org/en/stable/stats/
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats

from .io import filter_filename_token, filter_name_from_master_flat_path


def _load_data(path: Path) -> np.ndarray:
    return np.asarray(fits.getdata(path), dtype=float)


def image_stats(data: np.ndarray) -> dict[str, float]:
    med = float(np.nanmedian(data))
    mean = float(np.nanmean(data))
    std = float(np.nanstd(data))
    robust_mean, robust_med, robust_std = sigma_clipped_stats(data, sigma=5.0)
    return {
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": mean,
        "median": med,
        "std": std,
        "mad_std": float(mad_std(data)),
        "clipped_mean": float(robust_mean),
        "clipped_median": float(robust_med),
        "clipped_std": float(robust_std),
        "n_nan": int(np.isnan(data).sum()),
        "n_neg": int(np.sum(data < 0)),
        "frac_neg": float(np.mean(data < 0)),
    }


def format_stats(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name}\n"
        f"  min={stats['min']:.4g}  max={stats['max']:.4g}\n"
        f"  mean={stats['mean']:.4g}  median={stats['median']:.4g}\n"
        f"  std={stats['std']:.4g}  mad_std={stats['mad_std']:.4g}\n"
        f"  σ-clipped (5σ): mean={stats['clipped_mean']:.4g}  "
        f"median={stats['clipped_median']:.4g}  std={stats['clipped_std']:.4g}\n"
        f"  NaNs={stats['n_nan']}  negative pixels={stats['n_neg']} "
        f"({100*stats['frac_neg']:.2f}%)"
    )


def check_bias(stats: dict[str, float]) -> list[tuple[str, str]]:
    """Return list of (level, message) where level is ok/warn/fail.

    After overscan + combine (guide 02-04), master bias median should be ≈ 0 ADU;
    remaining structure is the 2-D pattern subtracted from flats/science.
    """
    notes = []
    # After overscan subtraction, bias should be near zero (guide 02-02 / 02-04)
    if abs(stats["median"]) < 5:
        notes.append(("ok", f"median ≈ 0 ({stats['median']:.3g} ADU) — overscan removal looks good"))
    else:
        notes.append(("warn", f"median is {stats['median']:.3g} ADU (expected ~0 after overscan)"))

    if stats["mad_std"] < 20:
        notes.append(("ok", f"robust scatter mad_std={stats['mad_std']:.3g} ADU (low read noise structure)"))
    else:
        notes.append(("warn", f"mad_std={stats['mad_std']:.3g} ADU looks high for a master bias"))

    if stats["n_nan"] == 0:
        notes.append(("ok", "no NaN pixels"))
    else:
        notes.append(("fail", f"{stats['n_nan']} NaN pixels"))
    return notes


def master_noise_notes(cfg: dict, master_path: Path, stats: dict[str, float]) -> list[tuple[str, str]]:
    """Compare the master's scatter with one raw bias (notebook rule: master < raw)."""
    from .frame_qc import check_master_noise, load_frame_section
    from .inventory import excluded_files, inventory_night

    obstype = str(cfg["obstypes"]["bias"]).upper()
    excluded = excluded_files(cfg)
    raw_path: Path | None = None
    for row in inventory_night(cfg)["rows"]:
        if row["obstype"] == obstype and row["file"] not in excluded:
            raw_path = Path(row["path"])
            break
    if raw_path is None:
        return []
    try:
        raw_data, _header = load_frame_section(raw_path, cfg)
        raw_mad = float(mad_std(raw_data[np.isfinite(raw_data)]))
        n_comb = fits.getheader(master_path, 0).get("NCOMBINE")
        n_comb = int(n_comb) if n_comb not in (None, "") else None
    except Exception:
        return []
    return check_master_noise(stats["mad_std"], raw_mad, n_comb)


def check_flat(stats: dict[str, float], *, filter_name: str) -> list[tuple[str, str]]:
    """OK/WARN notes for a master flat (guide 05-04: median ≈ 1 after inv-median)."""
    notes = []
    # inv_median scaling → combined flat median should be ~1
    # https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-04-Combining-flats.html
    if 0.95 <= stats["median"] <= 1.05:
        notes.append(("ok", f"median={stats['median']:.4f} ≈ 1 — scaling looks correct"))
    else:
        notes.append(
            ("warn", f"median={stats['median']:.4f} (expected ~1 after inv-median combine)")
        )

    if stats["mad_std"] < 0.2:
        notes.append(("ok", f"pixel-to-pixel variation mad_std={stats['mad_std']:.4f}"))
    else:
        notes.append(
            ("warn", f"large flat structure mad_std={stats['mad_std']:.4f} for filter {filter_name}")
        )

    if stats["frac_neg"] < 0.01:
        notes.append(("ok", f"negative fraction {100*stats['frac_neg']:.2f}% (small is OK near edges)"))
    else:
        notes.append(
            ("warn", f"negative fraction {100*stats['frac_neg']:.2f}% — check bias/dark subtraction")
        )

    if stats["min"] < -0.01:
        notes.append(
            ("warn", "flat has clearly negative pixels; inspect image before flat_correct")
        )
    elif stats["min"] <= 0:
        notes.append(
            ("ok", f"min={stats['min']:.3g} (≈0 numerical floor — usually fine)")
        )
    return notes


def _percentile_limits(data: np.ndarray, lo: float = 1, hi: float = 99) -> tuple[float, float]:
    return float(np.nanpercentile(data, lo)), float(np.nanpercentile(data, hi))


def parse_fits_section(sec: str) -> tuple[slice, slice]:
    """Parse FITS ``[x1:x2,y1:y2]`` (1-indexed, inclusive) → ``(row_slice, col_slice)``."""
    x1, x2, y1, y2 = section_bounds(sec)
    return slice(y1 - 1, y2), slice(x1 - 1, x2)


def section_bounds(sec: str) -> tuple[int, int, int, int]:
    """Return ``(x1, x2, y1, y2)`` from a FITS section, 1-indexed and inclusive."""
    text = str(sec).strip().strip("[]")
    xs, ys = text.split(",")
    x1, x2 = (int(v) for v in xs.split(":"))
    y1, y2 = (int(v) for v in ys.split(":"))
    return x1, x2, y1, y2


def format_fits_section(x1: int, x2: int, y1: int, y2: int) -> str:
    return f"[{int(x1)}:{int(x2)},{int(y1)}:{int(y2)}]"


def normalize_fits_section(sec: str) -> str:
    x1, x2, y1, y2 = section_bounds(sec)
    if x1 > x2 or y1 > y2:
        raise ValueError(f"empty FITS section: {sec}")
    return format_fits_section(x1, x2, y1, y2)


def section_inside(inner: str, outer: str) -> bool:
    """True when ``inner`` lies entirely inside ``outer``."""
    ix1, ix2, iy1, iy2 = section_bounds(inner)
    ox1, ox2, oy1, oy2 = section_bounds(outer)
    return ox1 <= ix1 <= ix2 <= ox2 and oy1 <= iy1 <= iy2 <= oy2


def overscan_side(biassec: str, datasec: str) -> str:
    """Which edge of the image the overscan sits on: right, left, high_y, low_y."""
    bx1, bx2, by1, by2 = section_bounds(biassec)
    dx1, dx2, dy1, dy2 = section_bounds(datasec)
    if bx1 > dx2:
        return "right"
    if bx2 < dx1:
        return "left"
    if by1 > dy2:
        return "high_y"
    if by2 < dy1:
        return "low_y"
    return "unknown"


def clip_section_y(sec: str, y_limits: str) -> str:
    """Keep ``sec`` x-range; clamp its y-range to ``y_limits``."""
    x1, x2, y1, y2 = section_bounds(sec)
    _a, _b, yy1, yy2 = section_bounds(y_limits)
    ny1, ny2 = max(y1, yy1), min(y2, yy2)
    if ny1 > ny2:
        raise ValueError(f"{sec} has no rows in common with {y_limits}")
    return format_fits_section(x1, x2, ny1, ny2)


def crop_section_rows(sec: str, *, n_lo: int = 0, n_hi: int = 0) -> str:
    """Drop ``n_lo`` rows from the start and ``n_hi`` from the end of a section."""
    if n_lo < 0 or n_hi < 0:
        raise ValueError("row crop cannot be negative")
    x1, x2, y1, y2 = section_bounds(sec)
    y1 += int(n_lo)
    y2 -= int(n_hi)
    if y1 > y2:
        raise ValueError("that row crop removes the whole section")
    return format_fits_section(x1, x2, y1, y2)


def row_settle_counts(row_med: np.ndarray, *, tol: float = 5.0) -> tuple[int, int, float]:
    """How many leading/trailing overscan rows sit off the typical level.

    Returns ``(n_lo, n_hi, typical)``. Typical is the median of the inner rows.
    This is readout settling at the chip end, not a slope along y.
    """
    values = np.asarray(row_med, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0, 0, float("nan")
    typical = float(np.median(finite))
    n_lo = 0
    for value in values:
        if not np.isfinite(value) or abs(float(value) - typical) > tol:
            n_lo += 1
        else:
            break
    n_hi = 0
    for value in values[::-1]:
        if not np.isfinite(value) or abs(float(value) - typical) > tol:
            n_hi += 1
        else:
            break
    if n_lo + n_hi >= int(values.size):
        return 0, 0, typical
    # A smooth slope is high at both ends relative to the middle. Settling
    # is a one-sided hook that a straight line cannot follow.
    if n_lo and n_hi and finite.size >= 4:
        x = np.arange(finite.size, dtype=float)
        resid = finite - np.polyval(np.polyfit(x, finite, 1), x)
        if float(np.ptp(resid)) < 20:
            return 0, 0, typical
    return n_lo, n_hi, typical


def suggest_row_crop(n_lo: int, n_rows: int, *, pad: int = 3, min_keep: int = 32) -> int:
    """Rows to drop from y=1, including a small pad after the last settling row."""
    if n_lo < 1:
        return 0
    keep = min(min_keep, max(1, n_rows // 2))
    return min(n_lo + pad, max(0, n_rows - keep))


def shrink_biassec(biassec: str, datasec: str, n_drop: int) -> str:
    """Drop ``n_drop`` pixels from the overscan edge that touches the image.

    Light leak enters from that edge. A right-side overscan ``[1025:1056,1:1024]``
    with ``n_drop=5`` becomes ``[1030:1056,1:1024]``.
    """
    if n_drop < 1:
        raise ValueError("drop at least one pixel")
    bx1, bx2, by1, by2 = section_bounds(biassec)
    side = overscan_side(biassec, datasec)
    if side == "right":
        bx1 += n_drop
    elif side == "left":
        bx2 -= n_drop
    elif side == "high_y":
        by1 += n_drop
    elif side == "low_y":
        by2 -= n_drop
    else:
        raise ValueError(
            f"cannot tell which edge of {biassec} touches the image {datasec}"
        )
    if bx1 > bx2 or by1 > by2:
        raise ValueError("that drop removes the whole overscan")
    return format_fits_section(bx1, bx2, by1, by2)


def overscan_edge_profile(
    data: np.ndarray, biassec: str, datasec: str
) -> tuple[str, list[tuple[int, float]]]:
    """Median profile of the overscan, ordered from the image edge outward.

    Returns ``(axis, pairs)`` where ``axis`` is ``column`` or ``row`` and each
    pair is ``(1-indexed coordinate, median ADU)``.
    """
    side = overscan_side(biassec, datasec)
    br, bc = parse_fits_section(biassec)
    over = np.asarray(data[br, bc], dtype=float)
    if side in ("right", "left", "unknown"):
        prof = np.nanmedian(over, axis=0)
        coords = range(bc.start + 1, bc.start + 1 + int(prof.size))
        pairs = [(int(c), float(v)) for c, v in zip(coords, prof)]
        if side == "left":
            pairs.reverse()
        return "column", pairs
    prof = np.nanmedian(over, axis=1)
    coords = range(br.start + 1, br.start + 1 + int(prof.size))
    pairs = [(int(c), float(v)) for c, v in zip(coords, prof)]
    if side == "low_y":
        pairs.reverse()
    return "row", pairs


def suggest_edge_drop(
    values_edge_first: list[float], *, tol: float = 5.0, min_keep: int = 8
) -> int:
    """How many image-edge pixels sit more than ``tol`` ADU above the far side.

    Stops at the first pixel within ``tol`` of the far-side median, and always
    leaves at least ``min_keep`` pixels (or half the region, if it is shorter).
    """
    n_pix = len(values_edge_first)
    if n_pix < 2:
        return 0
    keep = min_keep if n_pix > min_keep else max(1, n_pix // 2)
    limit = n_pix - keep
    if limit < 1:
        return 0
    far_n = max(1, n_pix // 4)
    far = float(np.median(values_edge_first[-far_n:]))
    n_drop = 0
    for value in values_edge_first[:limit]:
        if float(value) - far > tol:
            n_drop += 1
        else:
            break
    return n_drop


def _image_edge_level(data: np.ndarray, biassec: str, datasec: str, *, n: int = 8) -> float:
    """Median of the image edge that faces the overscan.

    Uses the last (or first) ``n`` pixels of ``DATASEC`` on the side next to
    the overscan. A trimmed ``BIASSEC`` leaves a gap; that gap is not the image.
    """
    side = overscan_side(biassec, datasec)
    dr, dc = parse_fits_section(datasec)
    if side == "right":
        start = max(dc.start, dc.stop - n)
        patch = data[:, start:dc.stop]
    elif side == "left":
        stop = min(dc.stop, dc.start + n)
        patch = data[:, dc.start:stop]
    elif side == "high_y":
        start = max(dr.start, dr.stop - n)
        patch = data[start:dr.stop, :]
    elif side == "low_y":
        stop = min(dr.stop, dr.start + n)
        patch = data[dr.start:stop, :]
    else:
        br, bc = parse_fits_section(biassec)
        start = max(dc.start, bc.start - n)
        patch = data[:, start:bc.start]
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _is_light_leak_note(level: str, msg: str) -> bool:
    if level == "ok":
        return False
    text = msg.lower()
    return "leak" in text or "edge" in text or "contamination" in text


def _load_raw_image(path: Path, cfg: dict) -> tuple[np.ndarray, dict]:
    """Load raw MEF/single-HDU image data + merged header (for BIASSEC/DATASEC)."""
    image_hdu = cfg["fits"]["image_hdu"]
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0].header.copy()
        if image_hdu in (0, "PRIMARY", None) or (
            isinstance(image_hdu, int) and image_hdu == 0
        ):
            data = np.asarray(hdul[0].data, dtype=float)
            header = primary
        else:
            ext = hdul[image_hdu]
            data = np.asarray(ext.data, dtype=float)
            header = primary
            header.update(ext.header)
    return data, header


def pick_raw_overscan_samples(cfg: dict) -> dict[str, Path]:
    """Pick one raw bias, flat, and science frame for overscan inspection."""
    from .inventory import inventory_night

    summary = inventory_night(cfg)
    obstypes = cfg["obstypes"]
    want = {
        "bias": obstypes["bias"].upper(),
        "flat": obstypes["flat"].upper(),
        "science": obstypes["science"].upper(),
    }
    found: dict[str, Path] = {}
    from .inventory import is_flat_obstype

    for kind, obstype in want.items():
        for row in summary["rows"]:
            matched = (
                is_flat_obstype(row["obstype"], obstype)
                if kind == "flat"
                else row["obstype"] == obstype
            )
            if matched:
                found[kind] = Path(row["path"])
                break
    return found


def analyze_overscan(data: np.ndarray, biassec: str, datasec: str) -> dict[str, Any]:
    """Measure overscan level, uniformity, and light-leak signatures.

    Guide 01-08 practice: column means into ``BIASSEC`` on flats most clearly
    show whether light has spilled into the nominal overscan. If contaminated,
    shrink the overscan section or stop trusting header ``BIASSEC``.
    https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-08-Overscan.html
    """
    br, bc = parse_fits_section(biassec)
    dr, dc = parse_fits_section(datasec)
    over = data[br, bc]
    useful = data[dr, dc]
    # Rows the fit will actually use: overscan ∩ trim. Settling rows that trim
    # will discard should not keep the y-warning alive after they are cropped.
    fit_start = max(br.start, dr.start)
    fit_stop = min(br.stop, dr.stop)
    over_fit = data[fit_start:fit_stop, bc] if fit_stop > fit_start else over

    mid = data.shape[0] // 2
    half = min(50, mid)
    band = np.nanmean(data[mid - half : mid + half, :], axis=0)

    oc0, oc1 = bc.start, bc.stop
    over_profile = band[oc0:oc1]
    q = max(1, over_profile.size // 4)
    first5 = float(np.nanmean(over_profile[: min(5, over_profile.size)]))
    last_n = min(20, over_profile.size)
    last20 = float(np.nanmean(over_profile[-last_n:]))
    left_q = float(np.nanmean(over_profile[:q]))
    right_q = float(np.nanmean(over_profile[-q:]))

    # Illuminated edge of DATASEC, not the pixels immediately before a trimmed
    # BIASSEC. Those gap columns are the ones a light-leak cut already rejected.
    useful_edge = _image_edge_level(data, biassec, datasec)

    row_med = np.nanmedian(over_fit, axis=1)
    over_finite = over[np.isfinite(over)]
    finite_rows = row_med[np.isfinite(row_med)]
    n_lo, n_hi, typical = row_settle_counts(row_med)
    inner = row_med
    if n_lo or n_hi:
        inner = row_med[n_lo : row_med.size - n_hi if n_hi else None]
    inner_finite = inner[np.isfinite(inner)]
    slope_adu = float("nan")
    resid_ptp = float("nan")
    if inner_finite.size >= 2:
        x = np.arange(inner_finite.size, dtype=float)
        coef = np.polyfit(x, inner_finite, 1)
        slope_adu = float(coef[0] * (inner_finite.size - 1))
        resid_ptp = float(np.ptp(inner_finite - np.polyval(coef, x)))

    return {
        "biassec": biassec,
        "datasec": datasec,
        "overscan_col0": oc0,
        "overscan_col1": oc1,
        "overscan_shape": tuple(over.shape),
        "median": float(np.nanmedian(over_finite)) if over_finite.size else float("nan"),
        "mad_std": float(mad_std(over_finite)) if over_finite.size else float("nan"),
        "useful_median": float(np.nanmedian(useful)),
        "useful_edge_median": useful_edge,
        "edge_excess": first5 - last20,  # >0 ⇒ elevated near science edge (leak)
        "left_minus_right": left_q - right_q,
        "row_median_ptp": float(np.ptp(finite_rows)) if finite_rows.size else float("nan"),
        "row_median_mad": float(mad_std(finite_rows)) if finite_rows.size else float("nan"),
        "row_medians": row_med,
        "y_settle_lo": int(n_lo),
        "y_settle_hi": int(n_hi),
        "y_typical": typical,
        "y_slope_adu": slope_adu,
        "y_resid_ptp": resid_ptp,
        "column_mean": band,
        "first5": first5,
        "last20": last20,
    }


def check_overscan(analysis: dict[str, Any], *, kind: str) -> list[tuple[str, str]]:
    """OK/WARN/FAIL notes for one raw-frame overscan analysis."""
    notes: list[tuple[str, str]] = []
    excess = analysis["edge_excess"]
    drop = analysis["useful_edge_median"] - analysis["median"]

    # Light leak into the nominal overscan (guide: flats show this most clearly)
    if kind == "flat":
        if abs(excess) < 5:
            notes.append(
                ("ok", f"little/no light leak into overscan (edge−far={excess:+.2f} ADU)")
            )
        elif abs(excess) < 30:
            notes.append(
                (
                    "warn",
                    f"mild overscan edge elevation {excess:+.2f} ADU — consider trimming "
                    "first overscan columns if persistent",
                )
            )
        else:
            notes.append(
                (
                    "fail",
                    f"strong light leak into BIASSEC (edge−far={excess:+.2f} ADU) — "
                    "header overscan region may be unsafe",
                )
            )
        if drop > 100:
            notes.append(
                ("ok", f"sharp drop from image edge into overscan (Δ={drop:.0f} ADU)")
            )
        else:
            notes.append(
                (
                    "warn",
                    f"weak drop from image edge into overscan (Δ={drop:.1f} ADU) — "
                    "possible contamination",
                )
            )
    else:
        if abs(excess) < 10:
            notes.append(("ok", f"overscan edge excess {excess:+.2f} ADU (low)"))
        else:
            notes.append(("warn", f"overscan edge excess {excess:+.2f} ADU"))

    # Spatial uniformity of the overscan itself
    if abs(analysis["left_minus_right"]) < 5:
        notes.append(
            (
                "ok",
                f"overscan fairly uniform along x "
                f"(left−right={analysis['left_minus_right']:+.2f} ADU)",
            )
        )
    else:
        notes.append(
            (
                "warn",
                f"overscan slope along x (left−right={analysis['left_minus_right']:+.2f} ADU)",
            )
        )

    n_lo = int(analysis.get("y_settle_lo") or 0)
    n_hi = int(analysis.get("y_settle_hi") or 0)
    bx1, bx2, by1, by2 = section_bounds(analysis["biassec"])
    if n_lo or n_hi:
        bits = []
        if n_lo:
            bits.append(f"first {n_lo} row(s) (FITS y={by1}…{by1 + n_lo - 1})")
        if n_hi:
            bits.append(f"last {n_hi} row(s)")
        y_start = by1 + n_lo
        y_end = by2 - n_hi
        notes.append(
            (
                "warn",
                "overscan "
                + " and ".join(bits)
                + f" still settling (ptp={analysis['row_median_ptp']:.0f} ADU, "
                f"typical={analysis.get('y_typical', float('nan')):.0f}) — "
                "not a slope the Legendre fit will remove. "
                f"Trim BIASSEC/DATASEC to y={y_start}:{y_end}",
            )
        )
    slope = analysis.get("y_slope_adu")
    resid = analysis.get("y_resid_ptp")
    if n_lo or n_hi:
        if resid == resid and resid < 20:
            notes.append(
                (
                    "ok",
                    f"after those rows, overscan is flat along y "
                    f"(line residual ptp={resid:.2f} ADU)",
                )
            )
        elif slope == slope and abs(slope) >= 20 and resid == resid and resid < 20:
            notes.append(
                (
                    "ok",
                    f"remaining y change is a smooth slope ({slope:+.1f} ADU); "
                    "Legendre-1 will subtract it",
                )
            )
    elif analysis["row_median_ptp"] < 20:
        notes.append(
            ("ok", f"overscan row medians stable (ptp={analysis['row_median_ptp']:.2f} ADU)")
        )
    elif slope == slope and resid == resid and resid < 20:
        notes.append(
            (
                "ok",
                f"overscan slope along y ({slope:+.1f} ADU over the chip); "
                "Legendre-1 will subtract it",
            )
        )
    else:
        notes.append(
            (
                "warn",
                f"overscan varies with row (ptp={analysis['row_median_ptp']:.2f} ADU)",
            )
        )

    notes.append(
        (
            "ok",
            f"overscan median={analysis['median']:.3f} ADU  "
            f"mad_std={analysis['mad_std']:.3f} ADU",
        )
    )
    return notes


def format_overscan_stats(label: str, analysis: dict[str, Any], path: Path) -> str:
    return (
        f"{label}  ({path.name})\n"
        f"  BIASSEC={analysis['biassec']}  DATASEC={analysis['datasec']}\n"
        f"  overscan median={analysis['median']:.3f}  mad_std={analysis['mad_std']:.3f}\n"
        f"  useful median={analysis['useful_median']:.3f}  "
        f"useful-edge median={analysis['useful_edge_median']:.3f}\n"
        f"  edge−far overscan={analysis['edge_excess']:+.3f} ADU  "
        f"left−right={analysis['left_minus_right']:+.3f} ADU  "
        f"row-median ptp={analysis['row_median_ptp']:.3f}  "
        f"y-settle lo/hi={analysis.get('y_settle_lo', 0)}/{analysis.get('y_settle_hi', 0)}"
    )


def label_panel(ax, *, title: str, xlabel: str, ylabel: str, ideal: str, width: int = 72) -> None:
    """Name the axes and say what a good panel looks like (wrapped to the panel)."""
    import textwrap

    lines = textwrap.wrap(title, width) + textwrap.wrap(f"Ideal: {ideal}", width)
    ax.set_title("\n".join(lines), fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def overscan_panels() -> list[dict[str, str]]:
    """Axis text for the raw-overscan figure (boundary, overscan-x, overscan-y)."""
    ylabel = "mean ADU in the central rows"
    xlabel = "column x (pixel, left to right)"
    return [
        {
            "title": "Central rows, from the image into the overscan",
            "xlabel": xlabel,
            "ylabel": ylabel,
            "ideal": "a sharp drop at the red line, then flat. A rise just inside the overscan is light leak.",
        },
        {
            "title": "Overscan columns only",
            "xlabel": xlabel,
            "ylabel": ylabel,
            "ideal": "each curve flat. Different heights are normal; a slope or a spike is not.",
        },
        {
            "title": "Overscan along y (bottom to top)",
            "xlabel": "row y (pixel, bottom to top)",
            "ylabel": "median ADU in that overscan row",
            "ideal": "flat after a few rows. A dip at y≈1 is readout settling — trim it; a slow tilt is a slope the fit can remove.",
        },
    ]


def diagnostic_panels(kind: str) -> list[dict[str, str]]:
    """Axis text for a master bias or master flat figure."""
    if kind == "bias":
        value = "ADU"
        cbar = "ADU"
        image_ideal = "smooth and near 0 ADU, with no bright edge"
        hist_ideal = "a narrow peak on 0 ADU"
        cut_ideal = "hug the orange 0 ADU line. Spikes are bad columns or rows."
    else:
        value = "relative response (1 = a typical pixel)"
        cbar = "relative response"
        image_ideal = "smooth and mostly near 1. Dust rings and vignetting are normal."
        hist_ideal = "a narrow peak on 1"
        cut_ideal = "follow the orange line at 1. A slow tilt is illumination; spikes are bad pixels."
    return [
        {
            "title": "Image (1–99% stretch). Dashed lines are the cuts below.",
            "xlabel": "column x (pixel)",
            "ylabel": "row y (pixel, bottom to top)",
            "ideal": image_ideal,
            "cbar": cbar,
        },
        {
            "title": "How many pixels have each value",
            "xlabel": f"pixel value ({value})",
            "ylabel": "number of pixels",
            "ideal": hist_ideal,
        },
        {
            "title": "Middle row, left to right",
            "xlabel": "column x (pixel, left to right)",
            "ylabel": f"{value} in that pixel",
            "ideal": cut_ideal,
        },
        {
            "title": "Middle column, bottom to top",
            "xlabel": "row y (pixel, bottom to top)",
            "ylabel": f"{value} in that pixel",
            "ideal": cut_ideal,
        },
    ]


def save_overscan_figure(
    samples: dict[str, dict[str, Any]],
    out_path: Path,
) -> Path:
    """
    Column-mean cuts into overscan for bias / flat / science.

    Left: region around the BIASSEC boundary. Right: overscan-only zoom.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {"bias": "C0", "flat": "C1", "science": "C2"}
    # Use first available sample to locate overscan columns
    any_sample = next(iter(samples.values()))
    oc0 = any_sample["analysis"]["overscan_col0"]
    oc1 = any_sample["analysis"]["overscan_col1"]
    has_y = any(
        rec.get("analysis", {}).get("row_medians") is not None for rec in samples.values()
    )
    n_panel = 3 if has_y else 2
    fig, axes = plt.subplots(1, n_panel, figsize=(6.2 * n_panel, 5.6), constrained_layout=True)
    if n_panel == 2:
        ax0, ax1 = axes
        ax2 = None
    else:
        ax0, ax1, ax2 = axes

    pad = 40
    x0 = max(0, oc0 - pad)
    notes = overscan_panels()

    for kind, rec in samples.items():
        band = rec["analysis"]["column_mean"]
        xs = np.arange(band.size)
        color = colors.get(kind, "k")
        ax0.plot(xs[x0:oc1], band[x0:oc1], color=color, lw=1.0, label=kind)
        ax1.plot(xs[oc0:oc1], band[oc0:oc1], color=color, lw=1.0, label=kind)
        row_med = rec["analysis"].get("row_medians")
        if ax2 is not None and row_med is not None:
            ys = np.arange(1, int(np.asarray(row_med).size) + 1)
            ax2.plot(ys, row_med, color=color, lw=1.0, label=kind)

    ax0.axvline(oc0, color="crimson", ls="--", lw=0.9, label="overscan starts")
    ax1.axvline(oc0, color="crimson", ls="--", lw=0.9, label="overscan starts")
    label_panel(ax0, **notes[0])
    label_panel(ax1, **notes[1])
    ax0.legend(fontsize=8)
    ax0.set_xlim(x0, max(oc1 - 1, x0 + 1))
    ax1.legend(fontsize=8)
    ax1.set_xlim(oc0, max(oc1 - 1, oc0 + 1))
    if ax2 is not None:
        label_panel(ax2, **notes[2])
        ax2.legend(fontsize=8)

    fig.suptitle("Raw overscan: light leak (x) and settling (y)?", fontsize=12)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def sanity_check_overscan(
    cfg: dict,
    *,
    diag_dir: Path,
) -> dict[str, Any]:
    """Inspect raw-frame overscan; write ``overscan_diag.png``."""
    from .calibrate import biassec_override, datasec_override
    from .term import dim, info, subheading, warn

    kw = cfg["keywords"]
    samples_paths = pick_raw_overscan_samples(cfg)
    result: dict[str, Any] = {
        "samples": {},
        "notes": [],
        "plot": None,
        "warnings": 0,
        "failures": 0,
        "light_leak": False,
        "y_settle": False,
        "y_settle_lo": 0,
        "y_settle_hi": 0,
        "header_biassec": None,
        "datasec": None,
        "header_datasec": None,
    }
    override = biassec_override(cfg)
    trim_override = datasec_override(cfg)

    subheading("Raw overscan (BIASSEC)")
    if not samples_paths:
        warn("  no raw frames found for overscan check")
        result["warnings"] += 1
        return result

    print(
        dim(
            "  Guide practice: check column means into overscan on bias + flat\n"
            "  (flats reveal light leak; frame-to-frame median shifts are expected).\n"
            "  See https://www.astropy.org/ccd-reduction-and-photometry-guide/"
            "v/dev/notebooks/01-08-Overscan.html"
        )
    )
    if override:
        print(dim(f"  Using config overscan.biassec={override} instead of the header."))
    if trim_override:
        print(dim(f"  Using config overscan.datasec={trim_override} instead of the header."))

    plot_samples: dict[str, dict[str, Any]] = {}
    all_notes: list[tuple[str, str]] = []

    for kind, path in samples_paths.items():
        data, header = _load_raw_image(path, cfg)
        header_biassec = header.get(kw["biassec"])
        datasec = header.get(kw["datasec"])
        if not header_biassec or not datasec:
            note = ("fail", f"{kind}: missing {kw['biassec']}/{kw['datasec']} in {path.name}")
            print(f"\n{kind}: {path.name}")
            _print_notes([note])
            all_notes.append(note)
            result["failures"] += 1
            continue

        header_biassec = str(header_biassec).strip()
        datasec = str(datasec).strip()
        biassec = override or header_biassec
        used_datasec = trim_override or datasec
        if result["header_biassec"] is None:
            result["header_biassec"] = header_biassec
            result["datasec"] = used_datasec
            result["header_datasec"] = datasec

        analysis = analyze_overscan(data, biassec, used_datasec)
        notes = check_overscan(analysis, kind=kind)
        print()
        print(format_overscan_stats(kind, analysis, path))
        _print_notes(notes)
        plot_samples[kind] = {"path": path, "analysis": analysis}
        result["y_settle_lo"] = max(result["y_settle_lo"], int(analysis.get("y_settle_lo") or 0))
        result["y_settle_hi"] = max(result["y_settle_hi"], int(analysis.get("y_settle_hi") or 0))
        result["samples"][kind] = {
            "path": str(path),
            "median": analysis["median"],
            "edge_excess": analysis["edge_excess"],
            "left_minus_right": analysis["left_minus_right"],
            "row_median_ptp": analysis["row_median_ptp"],
            "y_settle_lo": analysis.get("y_settle_lo", 0),
            "notes": notes,
        }
        all_notes.extend(notes)
        result["warnings"] += sum(1 for lvl, _ in notes if lvl == "warn")
        result["failures"] += sum(1 for lvl, _ in notes if lvl == "fail")

    # Cross-type pedestal shifts (informative — why overscan is useful)
    meds = {k: v["median"] for k, v in result["samples"].items()}
    if "bias" in meds and "flat" in meds:
        delta = meds["flat"] - meds["bias"]
        print(dim(f"\n  flat−bias overscan median = {delta:+.3f} ADU"))
    if "bias" in meds and "science" in meds:
        delta = meds["science"] - meds["bias"]
        print(dim(f"  science−bias overscan median = {delta:+.3f} ADU"))

    if plot_samples:
        plot = save_overscan_figure(plot_samples, diag_dir / "overscan_diag.png")
        result["plot"] = str(plot)
        info(f"  plot → {plot}")

    result["notes"] = all_notes
    result["light_leak"] = any(_is_light_leak_note(lvl, msg) for lvl, msg in all_notes)
    result["y_settle"] = result["y_settle_lo"] > 0 or result["y_settle_hi"] > 0
    return result


def row_column_profiles(
    data: np.ndarray,
    *,
    row: int | None = None,
    col: int | None = None,
) -> dict[str, Any]:
    """Extract middle (or chosen) row/column cuts and simple flatness metrics."""
    ny, nx = data.shape
    row = ny // 2 if row is None else int(row)
    col = nx // 2 if col is None else int(col)
    row = min(max(row, 0), ny - 1)
    col = min(max(col, 0), nx - 1)

    row_cut = np.asarray(data[row, :], dtype=float)
    col_cut = np.asarray(data[:, col], dtype=float)

    def _cut_stats(cut: np.ndarray) -> dict[str, float]:
        finite = cut[np.isfinite(cut)]
        if finite.size == 0:
            return {
                "median": float("nan"),
                "std": float("nan"),
                "ptp": float("nan"),
                "mad_std": float("nan"),
            }
        return {
            "median": float(np.median(finite)),
            "std": float(np.std(finite)),
            "ptp": float(np.ptp(finite)),
            "mad_std": float(mad_std(finite)),
        }

    return {
        "row_index": row,
        "col_index": col,
        "row_cut": row_cut,
        "col_cut": col_cut,
        "row_stats": _cut_stats(row_cut),
        "col_stats": _cut_stats(col_cut),
    }


def format_profile_stats(profiles: dict[str, Any]) -> str:
    rs, cs = profiles["row_stats"], profiles["col_stats"]
    return (
        f"  middle-row y={profiles['row_index']}: "
        f"median={rs['median']:.4g}  mad_std={rs['mad_std']:.4g}  "
        f"peak-to-peak={rs['ptp']:.4g}\n"
        f"  middle-col x={profiles['col_index']}: "
        f"median={cs['median']:.4g}  mad_std={cs['mad_std']:.4g}  "
        f"peak-to-peak={cs['ptp']:.4g}"
    )


def save_diagnostic_figure(
    data: np.ndarray,
    out_path: Path,
    *,
    title: str,
    kind: str,
    profiles: dict[str, Any] | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if profiles is None:
        profiles = row_column_profiles(data)

    row_i = profiles["row_index"]
    col_i = profiles["col_index"]
    row_cut = profiles["row_cut"]
    col_cut = profiles["col_cut"]

    # Downsample for image display if huge
    plot = data
    step = max(1, max(data.shape) // 1024)
    if step > 1:
        plot = data[::step, ::step]

    vmin, vmax = _percentile_limits(plot, 1, 99)
    ny, nx = data.shape
    panels = diagnostic_panels(kind)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    # --- image with cut markers ---
    ax_img = axes[0, 0]
    im = ax_img.imshow(
        plot,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
    )
    ax_img.axhline(row_i, color="crimson", ls="--", lw=0.9, alpha=0.9)
    ax_img.axvline(col_i, color="dodgerblue", ls="--", lw=0.9, alpha=0.9)
    label_panel(ax_img, **{k: v for k, v in panels[0].items() if k != "cbar"})
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046)
    cbar.set_label(panels[0]["cbar"])

    # --- histogram ---
    ax_hist = axes[0, 1]
    flat = data.ravel()
    if flat.size > 2_000_000:
        rng = np.random.default_rng(0)
        sample = rng.choice(flat, size=2_000_000, replace=False)
    else:
        sample = flat
    sample = sample[np.isfinite(sample)]
    ax_hist.hist(sample, bins=120, color="steelblue", alpha=0.85)
    ax_hist.axvline(
        np.median(sample), color="crimson", ls="--", label=f"median={np.median(sample):.4g}"
    )
    ideal = 0.0 if kind == "bias" else 1.0
    ax_hist.axvline(ideal, color="orange", ls=":", label=f"ideal level = {ideal:g}")
    label_panel(ax_hist, **panels[1])
    ax_hist.legend(fontsize=8)

    # Shared ADU scale for row/column cuts so structure is comparable
    cut_vals = np.concatenate(
        [
            np.asarray(row_cut, dtype=float).ravel(),
            np.asarray(col_cut, dtype=float).ravel(),
        ]
    )
    cut_vals = cut_vals[np.isfinite(cut_vals)]
    if cut_vals.size:
        y_lo, y_hi = np.percentile(cut_vals, [0.5, 99.5])
        if y_hi <= y_lo:
            y_lo, y_hi = float(np.min(cut_vals)), float(np.max(cut_vals))
        y_lo, y_hi = min(y_lo, ideal), max(y_hi, ideal)
        pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1.0
        shared_ylim = (y_lo - pad, y_hi + pad)
    else:
        shared_ylim = None

    # Shared pixel-axis extent when the detector is (nearly) square
    pix_max = max(row_cut.size, col_cut.size) - 1
    shared_xlim = (0, max(pix_max, 1))

    # --- middle-row profile (counts vs x) ---
    ax_row = axes[1, 0]
    x = np.arange(row_cut.size)
    ax_row.plot(x, row_cut, color="crimson", lw=0.7, label=f"row y={row_i}")
    ax_row.axhline(ideal, color="orange", ls=":", label=f"ideal level = {ideal:g}")
    ax_row.axhline(
        profiles["row_stats"]["median"],
        color="gray",
        ls="--",
        label=f"median={profiles['row_stats']['median']:.4g}",
    )
    label_panel(ax_row, **panels[2])
    ax_row.legend(fontsize=8)
    ax_row.set_xlim(*shared_xlim)
    if shared_ylim is not None:
        ax_row.set_ylim(*shared_ylim)

    # --- middle-column profile (counts vs y) ---
    ax_col = axes[1, 1]
    y = np.arange(col_cut.size)
    ax_col.plot(y, col_cut, color="dodgerblue", lw=0.7, label=f"column x={col_i}")
    ax_col.axhline(ideal, color="orange", ls=":", label=f"ideal level = {ideal:g}")
    ax_col.axhline(
        profiles["col_stats"]["median"],
        color="gray",
        ls="--",
        label=f"median={profiles['col_stats']['median']:.4g}",
    )
    label_panel(ax_col, **panels[3])
    ax_col.legend(fontsize=8)
    ax_col.set_xlim(*shared_xlim)
    if shared_ylim is not None:
        ax_col.set_ylim(*shared_ylim)

    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _print_notes(notes: list[tuple[str, str]]) -> None:
    from .term import status_tag

    for level, msg in notes:
        print(f"  {status_tag(level)} {msg}")


def _maybe_open(paths: list[Path]) -> None:
    if not paths or sys.platform != "darwin":
        return
    cmd = ["open", *[str(p) for p in paths]]
    try:
        from .term import show_command

        show_command(cmd)
        subprocess.run(cmd, check=False)
    except Exception:
        pass


def sanity_check_masters(
    cfg: dict,
    *,
    open_plots: bool = True,
    filters: list[str] | None = None,
    include_bias: bool = True,
    include_flats: bool = True,
) -> dict[str, Any]:
    """
    Run stats + diagnostic plots for overscan, master bias, and master flats.

    ``include_bias=False`` inspects only the raw overscan, so a light leak can
    be corrected before the master bias is built. ``include_flats=False`` stops
    after the master bias so a leak found later can be corrected before flats.

    Saves PNGs under ``<masters_dir>/diagnostics/``.
    """
    from .term import (
        S,
        banner,
        dim,
        fail,
        info,
        show_ccd,
        style,
        subheading,
        success,
        warn,
    )

    masters_dir = Path(cfg["paths"]["masters_dir"])
    diag_dir = masters_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "overscan": None,
        "bias": None,
        "flats": {},
        "plots": [],
        "warnings": 0,
        "failures": 0,
    }

    if not include_bias:
        banner("SANITY CHECK — raw overscan")
        scope = "the raw overscan before building a master bias."
        show_ccd(cfg, "sanity", "--overscan-only")
    elif include_flats:
        banner("SANITY CHECK — overscan + master calibration frames")
        scope = "masters (counts + image structure) before applying them to science."
        flat_flags: list[str] = ["--masters-only"]
        for filt_name in filters or []:
            flat_flags.extend(["--filter", filt_name])
        show_ccd(cfg, "sanity", *flat_flags)
    else:
        banner("SANITY CHECK — overscan + master bias")
        scope = "the master bias before combining flats."
        show_ccd(cfg, "sanity", "--bias-only")
    print(
        dim(
            "Checks follow CCD-guide practice: inspect raw overscan, then combined\n"
            f"{scope}\n"
        )
    )

    plot_paths: list[Path] = []

    overscan_report = sanity_check_overscan(cfg, diag_dir=diag_dir)
    report["overscan"] = overscan_report
    report["warnings"] += overscan_report["warnings"]
    report["failures"] += overscan_report["failures"]
    if overscan_report.get("plot"):
        plot_paths.append(Path(overscan_report["plot"]))

    bias_path = masters_dir / "master_bias.fits"

    if not include_bias:
        pass
    elif bias_path.exists():
        data = _load_data(bias_path)
        stats = image_stats(data)
        profiles = row_column_profiles(data)
        notes = check_bias(stats)
        notes.extend(master_noise_notes(cfg, bias_path, stats))
        subheading("Master bias")
        print(format_stats(bias_path.name, stats))
        print(dim(format_profile_stats(profiles)))
        _print_notes(notes)
        plot = save_diagnostic_figure(
            data,
            diag_dir / "master_bias_diag.png",
            title="Master bias",
            kind="bias",
            profiles=profiles,
        )
        plot_paths.append(plot)
        info(f"  plot → {plot}")
        report["bias"] = {
            "path": str(bias_path),
            "stats": stats,
            "notes": notes,
            "profiles": {
                "row_index": profiles["row_index"],
                "col_index": profiles["col_index"],
                "row_stats": profiles["row_stats"],
                "col_stats": profiles["col_stats"],
            },
        }
        report["warnings"] += sum(1 for lvl, _ in notes if lvl == "warn")
        report["failures"] += sum(1 for lvl, _ in notes if lvl == "fail")
    elif include_bias:
        subheading("Master bias")
        fail("  master_bias.fits not found")
        report["failures"] += 1

    flat_entries: list[tuple[Path, str]] = []
    if include_flats:
        flat_paths = sorted(masters_dir.glob("master_flat_*.fits"))
        for path in flat_paths:
            try:
                hdr = fits.getheader(path, 0)
            except Exception:
                hdr = None
            filt = filter_name_from_master_flat_path(path, hdr)
            flat_entries.append((path, filt))

        if filters is not None:
            wanted = set(filters)
            flat_entries = [(p, f) for p, f in flat_entries if f in wanted]

        if not flat_entries:
            subheading("Master flats")
            warn("  no master_flat_*.fits found")
            report["warnings"] += 1

    if flat_entries:
        subheading("Master flats")
        for path, filt in flat_entries:
            data = _load_data(path)
            stats = image_stats(data)
            profiles = row_column_profiles(data)
            notes = check_flat(stats, filter_name=filt)
            print()
            print(format_stats(f"{path.name}  (filter={filt})", stats))
            print(dim(format_profile_stats(profiles)))
            _print_notes(notes)
            plot = save_diagnostic_figure(
                data,
                diag_dir / f"master_flat_{filter_filename_token(filt)}_diag.png",
                title=f"Master flat '{filt}'",
                kind="flat",
                profiles=profiles,
            )
            plot_paths.append(plot)
            info(f"  plot → {plot}")
            report["flats"][filt] = {
                "path": str(path),
                "stats": stats,
                "notes": notes,
                "profiles": {
                    "row_index": profiles["row_index"],
                    "col_index": profiles["col_index"],
                    "row_stats": profiles["row_stats"],
                    "col_stats": profiles["col_stats"],
                },
            }
            report["warnings"] += sum(1 for lvl, _ in notes if lvl == "warn")
            report["failures"] += sum(1 for lvl, _ in notes if lvl == "fail")

    report["plots"] = [str(p) for p in plot_paths]

    print()
    print(style("-" * 60, S.DIM))
    if report["failures"]:
        summary_fn = fail
    elif report["warnings"]:
        summary_fn = warn
    else:
        summary_fn = success
    summary_fn(
        f"Summary: {report['failures']} failure(s), {report['warnings']} warning(s), "
        f"{len(plot_paths)} diagnostic plot(s)"
    )
    info(f"Diagnostics directory: {diag_dir}")
    if open_plots and plot_paths:
        print(dim("Opening diagnostic plots..."))
        _maybe_open(plot_paths)
    print(style("=" * 60, S.BOLD, S.CYAN))
    return report


def _format_edge_profile(
    axis: str, pairs: list[tuple[int, float]], *, suggested: int
) -> str:
    """Text table of the overscan, image edge first."""
    if not pairs:
        return "  (no overscan pixels)"
    far_n = max(1, len(pairs) // 4)
    far = float(np.median([v for _, v in pairs[-far_n:]]))
    show = pairs
    omitted = 0
    if len(pairs) > 48:
        show = pairs[:24] + pairs[-6:]
        omitted = len(pairs) - len(show)
    lines = [
        f"  {axis} median along the overscan, image edge → far side "
        f"(far-side level ≈ {far:.1f} ADU)",
    ]
    split_at = 24 if omitted else len(show)
    for i, (coord, value) in enumerate(show):
        if omitted and i == split_at:
            lines.append(f"  … {omitted} {axis}s omitted …")
        mark = "  leak" if value - far > 5 else ""
        lines.append(f"  {coord:6d}  {value:10.2f}{mark}")
    if suggested:
        lines.append(
            f"  Suggested: drop {suggested} {axis}(s) from the image edge."
        )
    else:
        lines.append("  No run of edge pixels sits more than 5 ADU above the far side.")
    return "\n".join(lines)


def _preview_overscan(cfg: dict, samples: dict, biassec: str, datasec: str) -> None:
    for kind, rec in samples.items():
        data, _header = _load_raw_image(Path(rec["path"]), cfg)
        analysis = analyze_overscan(data, biassec, datasec)
        notes = check_overscan(analysis, kind=kind)
        print()
        print(format_overscan_stats(kind, analysis, Path(rec["path"])))
        _print_notes(notes)


def offer_usable_overscan(
    cfg: dict,
    config_path: Path,
    report: dict[str, Any],
    *,
    open_plots: bool = True,
    rebuild_bias: bool = True,
    rebuild_flats: bool = True,
) -> dict[str, Any]:
    """After a light-leak warning, let the user pick the overscan section to keep.

    Writes ``overscan.biassec`` into the night YAML. Every later overscan
    subtraction for this night (rebuilt bias, flats, science) uses that section
    via ``resolve_biassec``. ``rebuild_bias=False`` only saves the region, for
    the check that runs before the master bias exists. ``rebuild_flats=False``
    rebuilds the bias only, for the check that runs before flats are combined.
    """
    if not sys.stdin.isatty():
        return report

    from .calibrate import biassec_override, datasec_override
    from .night_init import save_overscan_biassec
    from .prompt import ask, ask_int, ask_yes_no, choose
    from .term import dim, heading, info, warn

    while True:
        over = report.get("overscan") or {}
        leak = bool(over.get("light_leak"))
        n_lo = int(over.get("y_settle_lo") or 0)
        n_hi = int(over.get("y_settle_hi") or 0)
        if not leak and n_lo < 1 and n_hi < 1:
            return report
        header_biassec = over.get("header_biassec")
        datasec = over.get("header_datasec") or over.get("datasec")
        samples = over.get("samples") or {}
        if not header_biassec or not datasec or not samples:
            warn("Overscan needs a region change, but the sample frames have no BIASSEC to edit.")
            return report
        checked = biassec_override(cfg) or str(header_biassec)
        if n_lo or n_hi:
            prompt = (
                f"First {n_lo} overscan row(s) still settling "
                f"(not a slope). Drop them from overscan and trim?"
            )
            default_yes = True
        elif biassec_override(cfg):
            prompt = (
                f"Overscan {checked} still shows a light leak. Choose a different region?"
            )
            default_yes = False
        else:
            prompt = f"Light leak in overscan {checked}. Choose a usable region?"
            default_yes = True
        if not ask_yes_no(prompt, default_yes):
            return report

        heading("Usable overscan")
        current = biassec_override(cfg) or str(header_biassec)
        current_trim = datasec_override(cfg) or str(datasec)
        print(f"Header BIASSEC : {header_biassec}")
        print(f"Header trim    : {datasec}")
        print(f"Region checked : {current}")
        print(f"Trim checked   : {current_trim}")
        sample_kind = "flat" if "flat" in samples else next(iter(samples))
        sample = samples[sample_kind]
        data, _header = _load_raw_image(Path(sample["path"]), cfg)
        axis, pairs = overscan_edge_profile(data, current, str(current_trim))
        suggested = suggest_edge_drop([v for _, v in pairs])
        print(_format_edge_profile(axis, pairs, suggested=suggested))
        print(f"Profile from {sample_kind}: {Path(sample['path']).name}")
        if n_lo:
            ny = section_bounds(current)[3] - section_bounds(current)[2] + 1
            suggest_y = suggest_row_crop(n_lo, ny)
            print(
                dim(
                    f"  Y settling: first {n_lo} row(s). Suggested crop: "
                    f"drop y=1…{suggest_y} (pad {suggest_y - n_lo})."
                )
            )

        options: list[tuple[str, str]] = []
        if n_lo:
            ny = section_bounds(current)[3] - section_bounds(current)[2] + 1
            suggest_y = suggest_row_crop(n_lo, ny)
            options.append(
                (
                    "settle",
                    f"Drop first {suggest_y} rows (readout still settling)",
                )
            )
        options.extend(
            [
                ("drop", f"Drop {axis}s nearest the image"),
                ("section", "Enter a FITS section [x1:x2,y1:y2]"),
            ]
        )
        if biassec_override(cfg) or datasec_override(cfg):
            options.append(("header", "Use the header BIASSEC / trim again"))
        options.append(("keep", "Keep the region just checked"))
        choice = choose("Usable overscan:", options)
        if choice == "keep":
            return report

        clear_override = False
        new_trim = None
        if choice == "header":
            new_sec = str(header_biassec)
            new_trim = str(datasec)
            clear_override = True
        elif choice == "settle":
            ny = section_bounds(current_trim)[3] - section_bounds(current_trim)[2] + 1
            suggest_y = suggest_row_crop(n_lo, ny)
            n_drop = ask_int(
                f"Rows to drop from y=1 (1–{ny - 1})",
                suggest_y,
                minimum=1,
            )
            try:
                # Same rows leave the overscan and the trim, so the fit never
                # sees the settling samples.
                new_sec = crop_section_rows(current, n_lo=n_drop)
                new_trim = clip_section_y(
                    crop_section_rows(current_trim, n_lo=n_drop), new_sec
                )
            except ValueError as exc:
                warn(str(exc))
                continue
        elif choice == "drop":
            width = len(pairs)
            if width < 2:
                warn("Overscan is too narrow to trim.")
                continue
            n_drop = ask_int(
                f"{axis.capitalize()}s to drop from the image edge (1–{width - 1})",
                suggested or None,
                minimum=1,
            )
            if n_drop >= width:
                warn(f"Leave at least one {axis}.")
                continue
            try:
                new_sec = shrink_biassec(current, str(current_trim), n_drop)
            except ValueError as exc:
                warn(str(exc))
                continue
        else:
            raw = ask("FITS section")
            try:
                new_sec = normalize_fits_section(raw)
            except ValueError as exc:
                warn(str(exc))
                continue
            if not section_inside(new_sec, str(header_biassec)):
                warn(f"{new_sec} is outside the header overscan {header_biassec}.")
                continue
            try:
                new_trim = clip_section_y(current_trim, new_sec)
            except ValueError as exc:
                warn(str(exc))
                continue

        same_bias = new_sec == current and not (
            clear_override and biassec_override(cfg)
        )
        same_trim = new_trim is None or (
            new_trim == current_trim and not (clear_override and datasec_override(cfg))
        )
        if same_bias and same_trim:
            info(f"Already using {new_sec}.")
            return report

        print(f"\nCandidate overscan: {new_sec}")
        if new_trim:
            print(f"Candidate trim    : {new_trim}")
        _preview_overscan(cfg, samples, new_sec, str(new_trim or current_trim))
        if not ask_yes_no(f"Use {new_sec} for overscan subtraction?", True):
            continue

        save_overscan_biassec(
            config_path,
            cfg,
            None if clear_override else new_sec,
            datasec=None if clear_override else new_trim,
            update_datasec=bool(clear_override or new_trim),
        )
        info(f"Saved overscan region to {config_path}")
        info(
            "Bias, flats, and science for the rest of this night use this region."
        )
        if not rebuild_bias:
            info("Re-checking raw overscan with the new region.")
            report = sanity_check_masters(
                cfg,
                open_plots=open_plots,
                include_bias=False,
                include_flats=False,
            )
            continue

        rebuild_prompt = (
            "Rebuild master bias and flats with this overscan?"
            if rebuild_flats
            else "Rebuild master bias with this overscan?"
        )
        if not ask_yes_no(rebuild_prompt, True):
            warn(
                "The master bias on disk still used the previous overscan. "
                "Rebuild it before making flats or calibrating science."
            )
            return report

        from .calibrate import darks_enabled
        from .masters import make_master_bias, make_master_dark, make_master_flats

        make_master_bias(cfg)
        if darks_enabled(cfg):
            make_master_dark(cfg)
        if rebuild_flats:
            try:
                make_master_flats(cfg)
            except RuntimeError as exc:
                if "No flat frames" not in str(exc):
                    raise
                warn(str(exc))
        info("Re-checking overscan and master bias with the new region.")
        report = sanity_check_masters(
            cfg, open_plots=open_plots, include_flats=rebuild_flats
        )
    return report
