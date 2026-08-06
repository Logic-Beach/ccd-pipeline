"""Sanity checks and diagnostic plots for overscan + master bias / flats.

Inspired by Astropy CCD Reduction Guide practice:

* ``01-08-Overscan`` — inspect column means into ``BIASSEC`` (flats reveal
  light leak; decide whether the header overscan region is trustworthy)
* Master inspection — image display, histograms, and count statistics before
  applying calibrations to science (bias near 0 after overscan; flats ~1 after
  inv-median combine)

Guide: https://www.astropy.org/ccd-reduction-and-photometry-guide/
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats


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
    """Return list of (level, message) where level is ok/warn/fail."""
    notes = []
    # After overscan subtraction, bias should be near zero
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


def check_flat(stats: dict[str, float], *, filter_name: str) -> list[tuple[str, str]]:
    notes = []
    # inv_median scaling → combined flat median should be ~1
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
    sec = str(sec).strip().strip("[]")
    xs, ys = sec.split(",")
    x1, x2 = (int(v) for v in xs.split(":"))
    y1, y2 = (int(v) for v in ys.split(":"))
    return slice(y1 - 1, y2), slice(x1 - 1, x2)


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
    for kind, obstype in want.items():
        for row in summary["rows"]:
            if row["obstype"] == obstype:
                found[kind] = Path(row["path"])
                break
    return found


def analyze_overscan(data: np.ndarray, biassec: str, datasec: str) -> dict[str, Any]:
    """
    Measure overscan level, uniformity, and light-leak signatures.

    Follows the CCD-guide practice of inspecting a column-mean cut from the
    illuminated region into BIASSEC (flats are most sensitive to leak).
    """
    br, bc = parse_fits_section(biassec)
    dr, dc = parse_fits_section(datasec)
    over = data[br, bc]
    useful = data[dr, dc]

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

    # Columns just inside the illuminated edge (for sharp-drop check)
    edge_cols = band[max(oc0 - 8, dc.start or 0) : oc0]
    useful_edge = float(np.nanmedian(edge_cols)) if edge_cols.size else float("nan")

    row_med = np.nanmedian(over, axis=1)
    over_finite = over[np.isfinite(over)]

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
        "row_median_ptp": float(np.ptp(row_med[np.isfinite(row_med)])) if row_med.size else float("nan"),
        "row_median_mad": float(mad_std(row_med[np.isfinite(row_med)])) if row_med.size else float("nan"),
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
                ("ok", f"sharp drop at BIASSEC boundary (Δ={drop:.0f} ADU from useful edge)")
            )
        else:
            notes.append(
                (
                    "warn",
                    f"weak drop into overscan (Δ={drop:.1f} ADU) — possible contamination",
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

    if analysis["row_median_ptp"] < 20:
        notes.append(
            ("ok", f"overscan row medians stable (ptp={analysis['row_median_ptp']:.2f} ADU)")
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
        f"row-median ptp={analysis['row_median_ptp']:.3f}"
    )


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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax0, ax1 = axes
    pad = 40
    x0 = max(0, oc0 - pad)

    for kind, rec in samples.items():
        band = rec["analysis"]["column_mean"]
        xs = np.arange(band.size)
        color = colors.get(kind, "k")
        ax0.plot(xs[x0:oc1], band[x0:oc1], color=color, lw=1.0, label=kind)
        ax1.plot(xs[oc0:oc1], band[oc0:oc1], color=color, lw=1.0, label=kind)

    ax0.axvline(oc0, color="crimson", ls="--", lw=0.9, label="BIASSEC start")
    ax1.axvline(oc0, color="crimson", ls="--", lw=0.9)
    ax0.set_title("column mean near overscan boundary")
    ax0.set_xlabel("x (pixel)")
    ax0.set_ylabel("ADU (mid-row mean)")
    ax0.legend(fontsize=8)
    ax0.set_xlim(x0, oc1 - 1)

    ax1.set_title("overscan only (light-leak / uniformity check)")
    ax1.set_xlabel("x (pixel)")
    ax1.set_ylabel("ADU (mid-row mean)")
    ax1.legend(fontsize=8)
    ax1.set_xlim(oc0, oc1 - 1)

    fig.suptitle("Raw overscan inspection (CCD-guide style)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def sanity_check_overscan(
    cfg: dict,
    *,
    diag_dir: Path,
) -> dict[str, Any]:
    """Inspect raw-frame overscan; write ``overscan_diag.png``."""
    from .term import dim, info, subheading, warn

    kw = cfg["keywords"]
    samples_paths = pick_raw_overscan_samples(cfg)
    result: dict[str, Any] = {"samples": {}, "notes": [], "plot": None, "warnings": 0, "failures": 0}

    subheading("Raw overscan (BIASSEC)")
    if not samples_paths:
        warn("  no raw frames found for overscan check")
        result["warnings"] += 1
        return result

    print(
        dim(
            "Guide practice: check column means into overscan on bias + flat\n"
            "(flats reveal light leak; frame-to-frame median shifts are expected).\n"
        )
    )

    plot_samples: dict[str, dict[str, Any]] = {}
    all_notes: list[tuple[str, str]] = []

    for kind, path in samples_paths.items():
        data, header = _load_raw_image(path, cfg)
        biassec = header.get(kw["biassec"])
        datasec = header.get(kw["datasec"])
        if not biassec or not datasec:
            note = ("fail", f"{kind}: missing {kw['biassec']}/{kw['datasec']} in {path.name}")
            print(f"\n{kind}: {path.name}")
            _print_notes([note])
            all_notes.append(note)
            result["failures"] += 1
            continue

        analysis = analyze_overscan(data, str(biassec), str(datasec))
        notes = check_overscan(analysis, kind=kind)
        print()
        print(format_overscan_stats(kind, analysis, path))
        _print_notes(notes)
        plot_samples[kind] = {"path": path, "analysis": analysis}
        result["samples"][kind] = {
            "path": str(path),
            "median": analysis["median"],
            "edge_excess": analysis["edge_excess"],
            "left_minus_right": analysis["left_minus_right"],
            "row_median_ptp": analysis["row_median_ptp"],
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
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # --- image with cut markers ---
    ax_img = axes[0, 0]
    im = ax_img.imshow(plot, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    # Map full-res cut indices onto possibly downsampled display coords
    ax_img.axhline(row_i / step, color="crimson", ls="--", lw=0.9, alpha=0.9)
    ax_img.axvline(col_i / step, color="dodgerblue", ls="--", lw=0.9, alpha=0.9)
    ax_img.set_title("image (1–99% stretch)\nred=row cut  blue=col cut")
    ax_img.set_xlabel("x")
    ax_img.set_ylabel("y")
    fig.colorbar(im, ax=ax_img, fraction=0.046)

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
    if kind == "flat":
        ax_hist.axvline(1.0, color="orange", ls=":", label="ideal=1")
    elif kind == "bias":
        ax_hist.axvline(0.0, color="orange", ls=":", label="ideal=0")
    ax_hist.set_title("pixel value histogram")
    ax_hist.set_xlabel("ADU" if kind == "bias" else "relative response")
    ax_hist.set_ylabel("count")
    ax_hist.legend(fontsize=8)

    ylabel = "ADU" if kind == "bias" else "relative response"
    ideal = 0.0 if kind == "bias" else 1.0

    # --- middle-row profile (counts vs x) ---
    ax_row = axes[1, 0]
    x = np.arange(row_cut.size)
    ax_row.plot(x, row_cut, color="crimson", lw=0.7)
    ax_row.axhline(ideal, color="orange", ls=":", label=f"ideal={ideal:g}")
    ax_row.axhline(
        profiles["row_stats"]["median"],
        color="gray",
        ls="--",
        label=f"median={profiles['row_stats']['median']:.4g}",
    )
    ax_row.set_title(f"row cut  y={row_i}")
    ax_row.set_xlabel("x (pixel)")
    ax_row.set_ylabel(ylabel)
    ax_row.legend(fontsize=8)
    ax_row.set_xlim(0, max(row_cut.size - 1, 1))

    # --- middle-column profile (counts vs y) ---
    ax_col = axes[1, 1]
    y = np.arange(col_cut.size)
    ax_col.plot(y, col_cut, color="dodgerblue", lw=0.7)
    ax_col.axhline(ideal, color="orange", ls=":", label=f"ideal={ideal:g}")
    ax_col.axhline(
        profiles["col_stats"]["median"],
        color="gray",
        ls="--",
        label=f"median={profiles['col_stats']['median']:.4g}",
    )
    ax_col.set_title(f"column cut  x={col_i}")
    ax_col.set_xlabel("y (pixel)")
    ax_col.set_ylabel(ylabel)
    ax_col.legend(fontsize=8)
    ax_col.set_xlim(0, max(col_cut.size - 1, 1))

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
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
    try:
        subprocess.run(["open", *[str(p) for p in paths]], check=False)
    except Exception:
        pass


def sanity_check_masters(
    cfg: dict,
    *,
    open_plots: bool = True,
    filters: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run stats + diagnostic plots for master bias and flats.

    Saves PNGs under ``<masters_dir>/diagnostics/``.
    """
    from .term import (
        S,
        banner,
        dim,
        fail,
        info,
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

    banner("SANITY CHECK — overscan + master calibration frames")
    print(
        dim(
            "Checks follow CCD-guide practice: inspect raw overscan, then combined\n"
            "masters (counts + image structure) before applying them to science.\n"
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

    if bias_path.exists():
        data = _load_data(bias_path)
        stats = image_stats(data)
        profiles = row_column_profiles(data)
        notes = check_bias(stats)
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
    else:
        subheading("Master bias")
        fail("  master_bias.fits not found")
        report["failures"] += 1

    flat_paths = sorted(masters_dir.glob("master_flat_*.fits"))
    if filters is not None:
        wanted = set(filters)
        flat_paths = [p for p in flat_paths if p.stem.replace("master_flat_", "", 1) in wanted]

    if not flat_paths:
        subheading("Master flats")
        warn("  no master_flat_*.fits found")
        report["warnings"] += 1
    else:
        subheading("Master flats")
        for path in flat_paths:
            filt = path.stem.replace("master_flat_", "", 1)
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
                diag_dir / f"master_flat_{filt}_diag.png",
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
