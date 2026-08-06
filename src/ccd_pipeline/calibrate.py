"""Calibration primitives: overscan, trim, bias, flat."""

from __future__ import annotations

from astropy.nddata import CCDData
import ccdproc as ccdp
import numpy as np


def inv_median(array):
    return 1.0 / np.nanmedian(array)


def apply_overscan_and_trim(ccd: CCDData, cfg: dict) -> CCDData:
    """Subtract overscan using BIASSEC and trim to DATASEC."""
    kw = cfg["keywords"]
    biassec = ccd.meta.get(kw["biassec"])
    datasec = ccd.meta.get(kw["datasec"])
    if not biassec:
        raise ValueError("BIASSEC missing; cannot subtract overscan")
    if not datasec:
        raise ValueError("DATASEC missing; cannot trim")

    reduced = ccdp.subtract_overscan(
        ccd,
        fits_section=biassec,
        median=cfg.get("overscan", {}).get("median", True),
    )
    reduced = ccdp.trim_image(reduced, fits_section=datasec)
    return reduced


def subtract_bias(ccd: CCDData, master_bias: CCDData) -> CCDData:
    return ccdp.subtract_bias(ccd, master_bias)


def flat_correct(ccd: CCDData, master_flat: CCDData) -> CCDData:
    return ccdp.flat_correct(ccd, master_flat)


def combine_frames(paths_or_ccds, *, method: str = "average", scale=None,
                   sigma_clip: bool = True, sigma_clip_low: float = 5,
                   sigma_clip_high: float = 5, mem_limit: float = 4e9) -> CCDData:
    kwargs = dict(
        method=method,
        sigma_clip=sigma_clip,
        sigma_clip_low_thresh=sigma_clip_low if sigma_clip else None,
        sigma_clip_high_thresh=sigma_clip_high if sigma_clip else None,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=lambda a: 1.4826 * np.ma.median(np.abs(a - np.ma.median(a))),
        mem_limit=float(mem_limit),
    )
    # ccdproc.combine expects clip_func names differently across versions;
    # use mad_std when available.
    try:
        from astropy.stats import mad_std
        kwargs["sigma_clip_func"] = np.ma.median
        kwargs["sigma_clip_dev_func"] = mad_std
    except Exception:
        pass

    if scale is not None:
        kwargs["scale"] = scale

    combined = ccdp.combine(paths_or_ccds, **kwargs)
    combined.meta["combined"] = True
    return combined
