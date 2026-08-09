"""Calibration primitives: overscan, trim, bias, flat.

These steps mirror the Astropy CCD Reduction Guide / ``ccdproc`` workflow:

* Overscan + trim — notebooks ``01-08-Overscan``, ``02-02-Calibrating-bias-images``
* Master bias     — ``02-04-Combine-bias-images-to-make-master``
* Flats           — ``05-03-Calibrating-the-flats``, ``05-04-Combining-flats``
* Science         — ``06-00-Reducing-science-images``

Guide: https://www.astropy.org/ccd-reduction-and-photometry-guide/
"""

from __future__ import annotations

from astropy.nddata import CCDData
import ccdproc as ccdp
import numpy as np


def inv_median(array):
    """Scaling function for flat combination: weight ∝ 1 / median(frame).

    Used so brighter/fainter dome flats contribute equally before averaging
    (CCD guide combining-flats practice; ``ccdproc.combine(..., scale=...)``).
    """
    return 1.0 / np.nanmedian(array)


def describe_frame_steps(cfg: dict, *, kind: str) -> list[str]:
    """Human-readable plan of reduction steps for a frame type."""
    kw = cfg["keywords"]
    steps = [
        f"Load image HDU {cfg['fits']['image_hdu']!r} (unit={cfg['fits']['unit']})",
    ]
    if cfg.get("overscan", {}).get("median", True) is not False:
        how = "median" if cfg.get("overscan", {}).get("median", True) else "mean"
        steps.append(
            f"Subtract overscan from header {kw['biassec']} "
            f"(estimate={how} of overscan region)"
        )
    steps.append(f"Trim to useful area from header {kw['datasec']}")

    if kind == "bias":
        steps.append("Combine calibrated biases → master bias (σ-clip average)")
    elif kind == "flat":
        steps.append("Subtract master bias")
        if not cfg.get("darks", {}).get("enabled", False):
            steps.append("Dark subtraction: skipped (disabled / none for this night)")
        else:
            steps.append("Subtract master dark (scaled if needed)")
        steps.append("Combine flats with inverse-median scaling → master flat")
    elif kind == "science":
        steps.append("Subtract master bias")
        if not cfg.get("darks", {}).get("enabled", False):
            steps.append("Dark subtraction: skipped (disabled / none for this night)")
        else:
            steps.append("Subtract master dark (scaled if needed)")
        steps.append("Divide by matching master flat")
    return steps


def print_processing_plan(cfg: dict, *, kind: str, title: str | None = None) -> None:
    from .term import dim, heading

    heading(title or f"Processing plan ({kind})")
    print(dim("Processing plan:"))
    for i, step_txt in enumerate(describe_frame_steps(cfg, kind=kind), start=1):
        print(dim(f"  {i}. {step_txt}"))


def _append_history(ccd: CCDData, line: str) -> None:
    """Append a HISTORY card without wiping earlier pipeline notes.

    FITS commentary keywords are multi-card; never join lines with ``\\n``
    (astropy rejects non-printable characters in a single card value).
    """
    meta = ccd.meta
    if hasattr(meta, "add_history"):
        meta.add_history(line)
        return
    # Plain-dict meta (rare in tests): keep a list of discrete cards
    existing = meta.get("HISTORY")
    if existing is None:
        meta["HISTORY"] = [line]
    elif isinstance(existing, list):
        existing.append(line)
    else:
        meta["HISTORY"] = [str(existing), line]


def apply_overscan_and_trim(
    ccd: CCDData,
    cfg: dict,
    *,
    verbose: bool = False,
    label: str = "",
) -> CCDData:
    """Subtract overscan using ``BIASSEC``, then trim to ``DATASEC``.

    Bias frames are calibrated this way too (guide ``02-02``): overscan removes
    the per-frame pedestal so the master bias keeps residual 2-D structure.
    ``ccdproc.subtract_overscan`` with a side overscan uses a per-row median
    along the overscan columns (``overscan_axis=1`` by default).
    """
    kw = cfg["keywords"]
    biassec = ccd.meta.get(kw["biassec"])
    datasec = ccd.meta.get(kw["datasec"])
    if not biassec:
        raise ValueError("BIASSEC missing; cannot subtract overscan")
    if not datasec:
        raise ValueError("DATASEC missing; cannot trim")

    from .term import dim

    prefix = f"    [{label}] " if label else "    "
    shape0 = ccd.shape
    use_median = cfg.get("overscan", {}).get("median", True)

    if verbose:
        print(
            dim(
                f"{prefix}overscan subtract: section={biassec}  "
                f"estimate={'median' if use_median else 'mean'}"
            )
        )
    # Guide: ccdp.subtract_overscan(..., median=True) then trim_image
    reduced = ccdp.subtract_overscan(
        ccd,
        fits_section=biassec,
        median=use_median,
    )
    if verbose:
        print(dim(f"{prefix}trim: section={datasec}  shape {shape0} → "), end="")
    reduced = ccdp.trim_image(reduced, fits_section=datasec)
    if verbose:
        print(dim(f"{reduced.shape}"))

    how = "median" if use_median else "mean"
    _append_history(reduced, f"ccd_pipeline: overscan subtract {biassec} ({how})")
    _append_history(reduced, f"ccd_pipeline: trim to {datasec}")
    return reduced


def subtract_bias(
    ccd: CCDData,
    master_bias: CCDData,
    *,
    verbose: bool = False,
    label: str = "",
) -> CCDData:
    """Subtract the combined master bias (``ccdproc.subtract_bias``)."""
    from .term import dim

    prefix = f"    [{label}] " if label else "    "
    if verbose:
        print(dim(f"{prefix}subtract master bias"))
    out = ccdp.subtract_bias(ccd, master_bias)
    _append_history(out, "ccd_pipeline: subtract master bias")
    return out


def flat_correct(
    ccd: CCDData,
    master_flat: CCDData,
    *,
    verbose: bool = False,
    label: str = "",
) -> CCDData:
    """Divide by the master flat (``ccdproc.flat_correct``)."""
    from .term import dim

    prefix = f"    [{label}] " if label else "    "
    filt = master_flat.meta.get("FILTER", "?")
    if verbose:
        print(dim(f"{prefix}flat correct (master filter={filt})"))
    out = ccdp.flat_correct(ccd, master_flat)
    _append_history(out, f"ccd_pipeline: flat correct filter={filt}")
    return out


# Default combine budget: leave headroom on a ~64 GB machine; ccdproc's own
# default is 16 GB. Override per night with combine.*.mem_limit if needed.
DEFAULT_MEM_LIMIT = 48e9


def combine_frames(
    paths_or_ccds,
    *,
    method: str = "average",
    scale=None,
    sigma_clip: bool = True,
    sigma_clip_low: float = 5,
    sigma_clip_high: float = 5,
    mem_limit: float | None = None,
    verbose: bool = False,
) -> CCDData:
    """σ-clip combine via ``ccdproc.combine`` (guide image-combination practice)."""
    if mem_limit is None:
        mem_limit = DEFAULT_MEM_LIMIT
    kwargs = dict(
        method=method,
        sigma_clip=sigma_clip,
        sigma_clip_low_thresh=sigma_clip_low if sigma_clip else None,
        sigma_clip_high_thresh=sigma_clip_high if sigma_clip else None,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=lambda a: 1.4826 * np.ma.median(np.abs(a - np.ma.median(a))),
        mem_limit=float(mem_limit),
    )
    try:
        from astropy.stats import mad_std

        kwargs["sigma_clip_func"] = np.ma.median
        kwargs["sigma_clip_dev_func"] = mad_std
    except Exception:
        pass

    if scale is not None:
        kwargs["scale"] = scale

    n_frames = len(paths_or_ccds) if hasattr(paths_or_ccds, "__len__") else None
    if verbose:
        from .term import dim, run_with_spinner

        scale_txt = "inv_median" if scale is not None else "none"
        n_txt = f"{n_frames} frames, " if n_frames is not None else ""
        print(
            dim(
                f"    combine: {n_txt}method={method}  sigma_clip={sigma_clip} "
                f"({sigma_clip_low}/{sigma_clip_high})  scale={scale_txt}"
            )
        )
        combined = run_with_spinner(
            "combining (σ-clip average of large CCD frames)",
            ccdp.combine,
            paths_or_ccds,
            **kwargs,
        )
    else:
        combined = ccdp.combine(paths_or_ccds, **kwargs)

    combined.meta["combined"] = True
    return combined
