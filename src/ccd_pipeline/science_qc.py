"""Post-reduce quality checks on calibrated science frames.

Measures sky background level, scatter, and large-scale flatness (gradient /
row–column cuts) after bias+flat calibration. Complements master-frame sanity
checks in ``sanity.py`` (README Methodology §4).

Why this matters
----------------
After guide 06-00 calibration (overscan → bias → flat), residual large-scale
structure in the sky — especially in SDSS **i** / **z** — often indicates
fringing (additive airglow interference). High peak-to-peak in a coarse sky
map is a cue to run ``ccd fringe`` (see README §6; not covered by the Astropy
guide).

Sky stats use Astropy ``sigma_clipped_stats`` / ``mad_std``:
https://docs.astropy.org/en/stable/stats/

Science calibration context:
https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/06-00-Reducing-science-images.html
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats

from .term import S, dim, info, show_ccd, status_tag, style, subheading, success, warn
from .wcs_solve import iter_science_fits


# Filenames: [p|pf|pw|pfw-]night.object.filter.fits or …filter.NN.fits
def parse_science_filename(path: Path) -> dict[str, str]:
    """Parse ``{stages}-{night}.{object}.{filter}[.seq]`` science product names."""
    from .reduce import split_science_stage_name

    stages, rest = split_science_stage_name(Path(path).name)
    rest_path = Path(rest)
    parts = rest_path.stem.split(".")
    if len(parts) < 3:
        return {
            "night": "",
            "object": rest_path.stem,
            "filter": "",
            "seq": "",
            "stages": stages,
        }
    night = parts[0]
    if parts[-1].isdigit() and len(parts) >= 4:
        seq = parts[-1]
        filt = parts[-2]
        obj = ".".join(parts[1:-2])
    else:
        seq = ""
        filt = parts[-1]
        obj = ".".join(parts[1:-1])
    return {
        "night": night,
        "object": obj,
        "filter": filt,
        "seq": seq,
        "stages": stages,
    }


def group_science_by_filter(paths: list[Path]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for path in paths:
        filt = parse_science_filename(path).get("filter") or "unknown"
        out.setdefault(filt, []).append(path)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _sky_mask(data: np.ndarray, *, nsigma: float = 5.0) -> np.ndarray:
    """True for sky-like pixels (reject bright sources via σ-clip)."""
    _, med, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    if not np.isfinite(std) or std <= 0:
        std = float(mad_std(data)) or 1.0
    return np.isfinite(data) & (data < med + nsigma * std)


BAND_FRACTIONS = (0.25, 0.5, 0.75)


def band_cuts(
    data: np.ndarray, *, half: int = 25, fractions: tuple[float, ...] = BAND_FRACTIONS
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Median row/column bands at several heights/widths (IRAF ``implot :l a b``).

    Returns ``(row_cuts, col_cuts)`` keyed by a label such as ``"rows 487–537"``:
    each row cut is the median over a band of rows plotted against x; each column
    cut is the median over a band of columns plotted against y. Comparing bands
    from different parts of the chip shows whether a tilt or ripple is global.
    """
    ny, nx = data.shape
    row_cuts: dict[str, np.ndarray] = {}
    col_cuts: dict[str, np.ndarray] = {}
    for frac in fractions:
        y0 = int(round(frac * (ny - 1)))
        ya, yb = max(0, y0 - half), min(ny, y0 + half + 1)
        row_cuts[f"rows {ya}–{yb - 1}"] = np.nanmedian(data[ya:yb, :], axis=0)
        x0 = int(round(frac * (nx - 1)))
        xa, xb = max(0, x0 - half), min(nx, x0 + half + 1)
        col_cuts[f"cols {xa}–{xb - 1}"] = np.nanmedian(data[:, xa:xb], axis=1)
    return row_cuts, col_cuts


def analyze_science_sky(data: np.ndarray) -> dict[str, Any]:
    """Sky statistics + simple flatness metrics on a calibrated frame.

    Expectation after bias+flat (guide 06-00): sky is roughly flat aside from
    real gradients / fringing. Elevated coarse-map peak-to-peak in i/z is a
    fringe indicator (README Methodology §4 / §6).
    """
    data = np.asarray(data, dtype=float)
    sky = _sky_mask(data)
    sample = data[sky]
    if sample.size == 0:
        sample = data[np.isfinite(data)]

    med = float(np.nanmedian(sample))
    # Astropy robust sky: https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html
    robust_mean, robust_med, robust_std = sigma_clipped_stats(sample, sigma=5.0)
    sky_mad = float(mad_std(sample)) if sample.size else float("nan")

    # Large-scale background from a coarse binned map (source-resistant median).
    # High ptp here after flat-fielding often flags fringe / illumination issues.
    block = max(32, min(data.shape) // 64)
    ny, nx = data.shape
    by = max(1, ny // block)
    bx = max(1, nx // block)
    coarse = np.full((by, bx), np.nan)
    for iy in range(by):
        for ix in range(bx):
            tile = data[iy * block : (iy + 1) * block, ix * block : (ix + 1) * block]
            if tile.size:
                coarse[iy, ix] = np.nanmedian(tile)
    finite = coarse[np.isfinite(coarse)]
    sky_ptp = float(np.ptp(finite)) if finite.size else float("nan")
    sky_rms = float(np.nanstd(finite)) if finite.size else float("nan")
    frac_ptp = sky_ptp / abs(med) if abs(med) > 1e-6 else float("nan")

    # Corner means for a quick gradient sense
    h, w = max(1, ny // 8), max(1, nx // 8)
    corners = {
        "ll": float(np.nanmedian(data[:h, :w])),
        "lr": float(np.nanmedian(data[:h, -w:])),
        "ul": float(np.nanmedian(data[-h:, :w])),
        "ur": float(np.nanmedian(data[-h:, -w:])),
    }
    corner_ptp = float(np.ptp(list(corners.values())))

    mid = ny // 2
    half = min(25, mid)
    row_cut = np.nanmedian(data[mid - half : mid + half, :], axis=0)
    col_cut = np.nanmedian(data[:, nx // 2 - half : nx // 2 + half], axis=1)
    row_cuts, col_cuts = band_cuts(data, half=half)

    return {
        "row_cuts": row_cuts,
        "col_cuts": col_cuts,
        "sky_median": med,
        "sky_clipped_mean": float(robust_mean),
        "sky_clipped_std": float(robust_std),
        "sky_mad_std": sky_mad,
        "sky_map_ptp": sky_ptp,
        "sky_map_rms": sky_rms,
        "sky_ptp_frac": frac_ptp,
        "corners": corners,
        "corner_ptp": corner_ptp,
        "row_cut": row_cut,
        "col_cut": col_cut,
        "n_sky_pix": int(sky.sum()),
        "frac_sky": float(sky.mean()),
    }


def check_science_sky(stats: dict[str, Any], *, filter_name: str) -> list[tuple[str, str]]:
    notes: list[tuple[str, str]] = []
    med = stats["sky_median"]
    if np.isfinite(med):
        notes.append(("ok", f"sky median={med:.4g} ADU"))
    else:
        notes.append(("fail", "sky median is not finite"))

    frac = stats["sky_ptp_frac"]
    if np.isfinite(frac):
        pct = 100.0 * frac
        if pct < 3:
            notes.append(("ok", f"large-scale sky ptp={pct:.2f}% of median (quite flat)"))
        elif pct < 8:
            notes.append(
                ("warn", f"large-scale sky ptp={pct:.2f}% of median (mild gradient/structure)")
            )
        else:
            notes.append(
                (
                    "warn",
                    f"large-scale sky ptp={pct:.2f}% of median — check flat / fringe "
                    f"(esp. filter '{filter_name}')",
                )
            )

    if stats["sky_mad_std"] < max(5.0, 0.2 * abs(med) if abs(med) > 1 else 5.0):
        notes.append(("ok", f"sky scatter mad_std={stats['sky_mad_std']:.4g} ADU"))
    else:
        notes.append(("warn", f"sky scatter mad_std={stats['sky_mad_std']:.4g} ADU looks high"))

    if stats["frac_sky"] > 0.5:
        notes.append(("ok", f"sky-like pixels {100*stats['frac_sky']:.0f}% of frame"))
    else:
        notes.append(
            ("warn", f"only {100*stats['frac_sky']:.0f}% sky-like pixels — crowded field?")
        )
    return notes


def science_sky_panels() -> list[dict[str, str]]:
    """Axis text for a calibrated-science sky figure."""
    cut_ideal = (
        "all three bands flat on the orange sky level and on top of each other. "
        "Stars spike up; a shared tilt is a gradient; bands that disagree mean the flat is off."
    )
    return [
        {
            "title": "Calibrated frame (1–99% stretch)",
            "xlabel": "column x (pixel)",
            "ylabel": "row y (pixel, bottom to top)",
            "ideal": "even sky. Ripples in i or z are fringing; a corner-to-corner shade is the flat.",
        },
        {
            "title": "Sky pixels only, zoomed on the peak",
            "xlabel": "sky brightness (ADU)",
            "ylabel": "number of sky pixels",
            "ideal": "one narrow peak at the dashed sky median",
        },
        {
            "title": "Sky along row bands at 1/4, 1/2, 3/4 height (implot :l)",
            "xlabel": "column x (pixel, left to right)",
            "ylabel": "ADU (median of the band's rows)",
            "ideal": cut_ideal,
        },
        {
            "title": "Sky along column bands at 1/4, 1/2, 3/4 width (implot :c)",
            "xlabel": "row y (pixel, bottom to top)",
            "ylabel": "ADU (median of the band's columns)",
            "ideal": cut_ideal,
        },
    ]


def save_science_sky_figure(
    data: np.ndarray,
    stats: dict[str, Any],
    out_path: Path,
    *,
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    step = max(1, max(data.shape) // 1024)
    plot = data[::step, ::step]
    lo, hi = np.nanpercentile(plot, [1, 99])

    from .sanity import label_panel

    ny, nx = data.shape
    panels = science_sky_panels()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ax_img, ax_hist, ax_row, ax_col = axes.ravel()

    im = ax_img.imshow(
        plot,
        origin="lower",
        cmap="gray",
        vmin=lo,
        vmax=hi,
        extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
    )
    label_panel(ax_img, **panels[0])
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046)
    cbar.set_label("ADU")

    # Histogram of *sky* pixels only, zoomed around the sky peak.
    # Full-frame histograms are dominated by a narrow sky spike plus a long
    # empty tail to saturated stars (~65k ADU) — useless for QC.
    sky = _sky_mask(data)
    sample = data[sky]
    if sample.size == 0:
        sample = data[np.isfinite(data)]
    sample = sample[np.isfinite(sample)]
    if sample.size > 1_500_000:
        rng = np.random.default_rng(0)
        sample = rng.choice(sample, size=1_500_000, replace=False)

    med = float(stats.get("sky_median", np.nanmedian(sample)))
    mad = float(stats.get("sky_mad_std", mad_std(sample))) or 1.0
    # Window: sky median ± 8 robust σ, but at least the 1–99% of sky pixels
    p_lo, p_hi = np.nanpercentile(sample, [1, 99])
    x_lo = min(p_lo, med - 8.0 * mad)
    x_hi = max(p_hi, med + 8.0 * mad)
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        x_lo, x_hi = float(np.nanmin(sample)), float(np.nanmax(sample))

    in_win = sample[(sample >= x_lo) & (sample <= x_hi)]
    ax_hist.hist(in_win, bins=100, color="steelblue", alpha=0.85, range=(x_lo, x_hi))
    ax_hist.axvline(med, color="crimson", ls="--", label=f"sky median={med:.4g}")
    ax_hist.set_xlim(x_lo, x_hi)
    label_panel(ax_hist, **panels[1])
    ax_hist.legend(fontsize=8)

    row_cuts = stats.get("row_cuts") or {"middle rows": stats["row_cut"]}
    col_cuts = stats.get("col_cuts") or {"middle columns": stats["col_cut"]}
    band_colors = ("mediumseagreen", "crimson", "mediumpurple")
    for (label, cut), color in zip(row_cuts.items(), band_colors):
        ax_row.plot(cut, color=color, lw=0.7, label=label)
    ax_row.axhline(stats["sky_median"], color="orange", ls=":", label="sky median")
    label_panel(ax_row, **panels[2])
    ax_row.legend(fontsize=8)

    band_colors = ("mediumseagreen", "dodgerblue", "mediumpurple")
    for (label, cut), color in zip(col_cuts.items(), band_colors):
        ax_col.plot(cut, color=color, lw=0.7, label=label)
    ax_col.axhline(stats["sky_median"], color="orange", ls=":", label="sky median")
    label_panel(ax_col, **panels[3])
    ax_col.legend(fontsize=8)

    # Zoom both cut panels on the sky so a few-percent tilt is visible under the stars.
    med = float(stats["sky_median"])
    mad = float(stats.get("sky_mad_std", 0.0) or 0.0)
    if np.isfinite(med) and mad > 0:
        for ax in (ax_row, ax_col):
            ax.set_ylim(med - 6 * mad, med + 10 * mad)

    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


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


def sanity_check_science(
    cfg: dict,
    *,
    filters: list[str] | None = None,
    limit_per_filter: int = 2,
    prefer_blank: bool = True,
    open_plots: bool = True,
) -> dict[str, Any]:
    """Inspect a few calibrated science frames per filter for sky flatness."""
    from .term import banner

    paths = iter_science_fits(cfg)
    by_filt = group_science_by_filter(paths)
    if filters is not None:
        want = set(filters)
        by_filt = {k: v for k, v in by_filt.items() if k in want}

    out_dir = Path(cfg["paths"]["output_dir"])
    diag_dir = out_dir / "science_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "filters": {},
        "plots": [],
        "warnings": 0,
        "failures": 0,
    }

    banner("SANITY CHECK — calibrated science sky")
    science_flags = ["--science-only", "--science-limit", str(limit_per_filter)]
    for filt_name in filters or []:
        science_flags.extend(["--filter", filt_name])
    show_ccd(cfg, "sanity", *science_flags)
    print(
        dim(
            "Sky level / flatness after bias+flat. Large structure in z/i often\n"
            "means fringing (additive sky-line interference) — see ccd fringe.\n"
        )
    )

    if not by_filt:
        warn("No calibrated science FITS found — run ccd reduce first")
        report["warnings"] += 1
        return report

    plot_paths: list[Path] = []
    blank_keys = ("blank", "sky", "empty")

    for filt, flist in by_filt.items():
        subheading(f"filter '{filt}'")
        ranked = list(flist)
        if prefer_blank:
            blanks = [
                p
                for p in ranked
                if any(k in parse_science_filename(p)["object"].lower() for k in blank_keys)
            ]
            others = [p for p in ranked if p not in blanks]
            ranked = blanks + others
        sample = ranked[: max(1, int(limit_per_filter))]
        report["filters"][filt] = []

        for path in sample:
            data = np.asarray(fits.getdata(path), dtype=float)
            stats = analyze_science_sky(data)
            notes = check_science_sky(stats, filter_name=filt)
            meta = parse_science_filename(path)
            print()
            print(style(path.name, S.BOLD))
            print(
                dim(
                    f"  OBJECT={meta['object']!r}  sky_median={stats['sky_median']:.4g}  "
                    f"mad_std={stats['sky_mad_std']:.4g}  "
                    f"sky-map ptp={stats['sky_map_ptp']:.4g} "
                    f"({100*stats['sky_ptp_frac']:.2f}% of median)  "
                    f"corner_ptp={stats['corner_ptp']:.4g}"
                )
            )
            for level, msg in notes:
                print(f"  {status_tag(level)} {msg}")
            plot = save_science_sky_figure(
                data,
                stats,
                diag_dir / f"{path.stem}_sky.png",
                title=f"Science sky — {path.name}",
            )
            plot_paths.append(plot)
            info(f"  plot → {plot}")
            report["filters"][filt].append(
                {
                    "path": str(path),
                    "stats": {
                        k: v
                        for k, v in stats.items()
                        if k not in {"row_cut", "col_cut", "row_cuts", "col_cuts", "corners"}
                    },
                    "notes": notes,
                }
            )
            report["warnings"] += sum(1 for lvl, _ in notes if lvl == "warn")
            report["failures"] += sum(1 for lvl, _ in notes if lvl == "fail")

    report["plots"] = [str(p) for p in plot_paths]
    print()
    if report["failures"]:
        warn(
            f"Science sky summary: {report['failures']} failure(s), "
            f"{report['warnings']} warning(s)"
        )
    elif report["warnings"]:
        warn(
            f"Science sky summary: {report['warnings']} warning(s), "
            f"{len(plot_paths)} plot(s)"
        )
    else:
        success(
            f"Science sky summary: OK — {len(plot_paths)} diagnostic plot(s) in {diag_dir}"
        )
    if open_plots and plot_paths:
        _maybe_open(plot_paths)
    return report
