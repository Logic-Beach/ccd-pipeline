"""Calibration primitives: overscan, trim, bias, flat.

These steps mirror the Astropy CCD Reduction Guide / ``ccdproc`` workflow.
See package README “Methodology” and:

* Overscan + trim — https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-08-Overscan.html
  and https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/02-02-Calibrating-bias-images.html
* Master bias     — https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/02-04-Combine-bias-images-to-make-master.html
* Flats           — https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-03-Calibrating-the-flats.html
  and https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/05-04-Combining-flats.html
* Science         — https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/06-00-Reducing-science-images.html

``ccdproc`` API: https://ccdproc.readthedocs.io/
"""

from __future__ import annotations

from astropy.nddata import CCDData
import ccdproc as ccdp
import numpy as np

from . import GUIDE


def inv_median(array):
    """Scaling for flat combination: each frame × (1 / median).

    Twilight/dome flats change level frame-to-frame; without scaling, brighter
    frames dominate the stack. Guide practice is to normalize to a common
    median (often 1) before combining — see GUIDE['flat_combine'] and
    ``ccdproc.combine(..., scale=...)`` /
    https://ccdproc.readthedocs.io/en/stable/image_combination.html
    """
    return 1.0 / np.nanmedian(array)


def darks_enabled(cfg: dict) -> bool:
    """True when this night should subtract a master dark."""
    return bool((cfg.get("darks") or {}).get("enabled", False))


def exposure_seconds(meta, exptime_key: str = "EXPTIME") -> float | None:
    """Exposure time in seconds from a header, or ``None`` when it is missing.

    Zero or missing values fall through to the next keyword. ``DARKTIME`` is
    tried after ``EXPTIME``: KPNO/WIYN dark frames record ``EXPTIME = 0`` (the
    shutter never opened) and keep the real integration in ``DARKTIME``.
    """
    for key in (exptime_key, "EXPTIME", "DARKTIME", "EXPOSURE", "EXPTIM"):
        if meta is None or key not in meta or meta[key] in (None, ""):
            continue
        try:
            value = float(meta[key])
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def biassec_override(cfg: dict) -> str | None:
    """Return ``overscan.biassec`` when the night config replaces the header region."""
    raw = (cfg.get("overscan") or {}).get("biassec")
    if raw in (None, ""):
        return None
    return str(raw).strip()


def datasec_override(cfg: dict) -> str | None:
    """Return ``overscan.datasec`` when the night config replaces the trim region."""
    raw = (cfg.get("overscan") or {}).get("datasec")
    if raw in (None, ""):
        return None
    return str(raw).strip()


# Overscan fit models (IRAF ``ccdproc.function`` equivalents). ``none`` keeps
# ccdproc's raw per-row estimate, which is only right when the overscan is wide
# enough that a single row's median is quiet (see README "Overscan fitting").
OVERSCAN_MODELS = {
    "legendre": "Legendre1D",
    "chebyshev": "Chebyshev1D",
    "polynomial": "Polynomial1D",
}
DEFAULT_OVERSCAN_MODEL = "legendre"
DEFAULT_OVERSCAN_ORDER = 1


def overscan_settings(cfg: dict) -> dict:
    """Normalised ``overscan`` block: ``median``, ``model`` (or None), ``order``.

    ``model`` names the astropy.modeling polynomial fitted along the overscan
    (IRAF ``function=legendre``); ``order`` is its degree (0 = one constant
    per frame like IRAF ``order=1``; 1 = a straight line). ``model: none``
    subtracts each row's own median/mean, which prints the overscan read noise
    onto every row as horizontal stripes.
    """
    over = cfg.get("overscan") or {}
    median = over.get("median", True) is not False
    raw_model = over.get("model", DEFAULT_OVERSCAN_MODEL)
    model = None if raw_model in (None, "", "none", "None", False) else str(raw_model).lower()
    if model is not None and model not in OVERSCAN_MODELS:
        raise ValueError(
            f"overscan.model={raw_model!r} not supported; use one of "
            f"{', '.join(OVERSCAN_MODELS)} or none"
        )
    try:
        order = int(over.get("order", DEFAULT_OVERSCAN_ORDER))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"overscan.order must be an integer, got {over.get('order')!r}") from exc
    if order < 0:
        raise ValueError("overscan.order must be 0 or larger")
    return {"median": median, "model": model, "order": order}


def overscan_model(cfg: dict):
    """astropy.modeling instance for ``overscan.model``/``order`` (None = per row)."""
    settings = overscan_settings(cfg)
    if settings["model"] is None:
        return None
    from astropy.modeling import models

    cls = getattr(models, OVERSCAN_MODELS[settings["model"]])
    return cls(settings["order"])


def describe_overscan_estimate(cfg: dict) -> str:
    """Short text for plans/HISTORY: how the overscan level is estimated."""
    settings = overscan_settings(cfg)
    how = "median" if settings["median"] else "mean"
    if settings["model"] is None:
        return f"{how} of each row, subtracted row by row"
    degree = settings["order"]
    shape = "constant" if degree == 0 else ("straight line" if degree == 1 else f"degree-{degree} curve")
    return f"{how} of each row, then a {settings['model']} {shape} fitted along the rows"


def resolve_biassec(header, cfg: dict) -> str:
    """Overscan section to subtract: config override, otherwise the header keyword."""
    override = biassec_override(cfg)
    if override:
        return override
    kw = cfg["keywords"]["biassec"]
    value = header.get(kw)
    if not value:
        raise ValueError(f"{kw} missing; cannot subtract overscan")
    return str(value).strip()


def resolve_datasec(header, cfg: dict) -> str:
    """Trim section: config override, otherwise the header keyword."""
    override = datasec_override(cfg)
    if override:
        return override
    kw = cfg["keywords"]["datasec"]
    value = header.get(kw)
    if not value:
        raise ValueError(f"{kw} missing; cannot trim")
    return str(value).strip()


def describe_frame_steps(cfg: dict, *, kind: str) -> list[str]:
    """Human-readable plan of reduction steps for a frame type."""
    kw = cfg["keywords"]
    steps = [
        f"Load image HDU {cfg['fits']['image_hdu']!r} (unit={cfg['fits']['unit']})",
    ]
    override = biassec_override(cfg)
    if override:
        where = f"config overscan.biassec={override}"
    else:
        where = f"header {kw['biassec']}"
    steps.append(
        f"Subtract overscan from {where} (estimate: {describe_overscan_estimate(cfg)})"
    )
    trim_override = datasec_override(cfg)
    if trim_override:
        steps.append(f"Trim to config overscan.datasec={trim_override}")
    else:
        steps.append(f"Trim to useful area from header {kw['datasec']}")

    if kind == "bias":
        steps.append("Combine calibrated biases → master bias (σ-clip average)")
    elif kind == "dark":
        steps.append("Subtract master bias")
        steps.append("Scale each dark to a common exposure, then σ-clip average → master dark")
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


def _fits_section_expr(*, override: str | None, keyword: str) -> str:
    if override:
        return repr(override)
    return f"header[{keyword!r}]"


def _combine_example(cfg: dict, which: str, *, scale: str | None = None) -> str:
    combine = cfg.get("combine") or {}
    comb = combine.get(which) or {}
    if not comb and which == "dark":
        comb = combine.get("bias") or {}
    method = comb.get("method", "average")
    sigma = bool(comb.get("sigma_clip", True))
    parts = [f"method={method!r}", f"sigma_clip={sigma}"]
    if sigma:
        parts.append(f"sigma_clip_low_thresh={comb.get('sigma_clip_low', 5)}")
        parts.append(f"sigma_clip_high_thresh={comb.get('sigma_clip_high', 5)}")
    if scale:
        parts.append(f"scale={scale}")
    return "ccdproc.combine(frames, " + ", ".join(parts) + ")"


def astropy_examples(cfg: dict, *, kind: str) -> list[str]:
    """ccdproc calls this step actually runs, with this night's settings."""
    kw = cfg["keywords"]
    override = biassec_override(cfg)
    settings = overscan_settings(cfg)
    biassec = _fits_section_expr(override=override, keyword=kw["biassec"])
    datasec = _fits_section_expr(
        override=datasec_override(cfg), keyword=kw["datasec"]
    )
    model_arg = ""
    if settings["model"] is not None:
        model_arg = f", model=models.{OVERSCAN_MODELS[settings['model']]}({settings['order']})"
    lines = [
        "ccdproc.subtract_overscan(ccd, fits_section="
        f"{biassec}, median={settings['median']}{model_arg})",
        f"ccdproc.trim_image(ccd, fits_section={datasec})",
    ]
    exptime_key = (cfg.get("keywords") or {}).get("exptime", "EXPTIME")
    dark_call = (
        "ccdproc.subtract_dark(ccd, master_dark, "
        f"exposure_time={exptime_key!r}, exposure_unit=u.s, scale=True)"
    )
    if kind == "bias":
        lines.append(_combine_example(cfg, "bias"))
    elif kind == "dark":
        lines.append("ccdproc.subtract_bias(ccd, master_bias)")
        lines.append(_combine_example(cfg, "dark"))
    elif kind == "flat":
        lines.append("ccdproc.subtract_bias(ccd, master_bias)")
        if darks_enabled(cfg):
            lines.append(dark_call)
        flat = (cfg.get("combine") or {}).get("flat") or {}
        scale = "inv_median" if flat.get("scale") == "inv_median" else None
        lines.append(_combine_example(cfg, "flat", scale=scale))
    elif kind == "science":
        lines.append("ccdproc.subtract_bias(ccd, master_bias)")
        if darks_enabled(cfg):
            lines.append(dark_call)
        lines.append("ccdproc.flat_correct(ccd, master_flat)")
    return lines


def print_processing_plan(
    cfg: dict,
    *,
    kind: str,
    title: str | None = None,
    extra: list[str] | None = None,
) -> None:
    from .term import dim, heading, show_astropy, show_ccd

    heading(title or f"Processing plan ({kind})")
    flags = list(extra or [])
    if kind == "bias":
        show_ccd(cfg, "masters", "--only", "bias", *flags)
    elif kind == "dark":
        show_ccd(cfg, "masters", "--only", "darks", *flags)
    elif kind == "flat":
        show_ccd(cfg, "masters", "--only", "flats", *flags)
    elif kind == "science":
        show_ccd(cfg, "reduce", *flags)
    for call in astropy_examples(cfg, kind=kind):
        show_astropy(call)
    print(dim("Processing plan:"))
    for i, step_txt in enumerate(describe_frame_steps(cfg, kind=kind), start=1):
        print(dim(f"  {i}. {step_txt}"))


def _append_history(ccd: CCDData, line: str) -> None:
    """Append one FITS HISTORY card (never embed newlines).

    FITS commentary cards must be printable ASCII. Joining prior HISTORY with
    ``\\n`` into a single value raises ``ValueError`` in Astropy — use
    ``Header.add_history`` so each note is its own card.
    """
    # Strip non-printable / non-ASCII so header validation cannot fail.
    clean = "".join(ch if 32 <= ord(ch) <= 126 else "?" for ch in str(line))
    meta = ccd.meta
    if hasattr(meta, "add_history"):
        meta.add_history(clean)
        return
    existing = meta.get("HISTORY", None)
    if existing is None:
        meta["HISTORY"] = clean
    elif isinstance(existing, list):
        existing.append(clean)
        meta["HISTORY"] = existing
    else:
        # dict-like fallback: keep discrete entries, never newline-join
        meta["HISTORY"] = [str(existing), clean]


def _crop_to_biassec_rows(ccd: CCDData, biassec: str) -> tuple[CCDData, str, int]:
    """Drop rows outside ``biassec`` before the overscan is estimated.

    ``ccdproc.subtract_overscan`` fits every row of the array it is given, and
    the overscan section has to cover that whole array. Rows left out of
    ``BIASSEC`` are cut off first, so they are not samples in the Legendre
    fit or in the per-row median. No overscan pixel is replaced.

    Returns ``(cropped, biassec_on_crop, first_fits_row)``. ``first_fits_row``
    is the 1-indexed row of the original image that is row 1 of ``cropped``.
    """
    from .sanity import format_fits_section, section_bounds

    ny, nx = int(ccd.shape[0]), int(ccd.shape[1])
    bx1, bx2, by1, by2 = section_bounds(biassec)
    by1 = max(1, by1)
    by2 = min(ny, by2)
    if by1 > by2 or bx1 > bx2:
        raise ValueError(f"overscan {biassec} does not overlap this image")
    oscan = format_fits_section(bx1, bx2, 1, by2 - by1 + 1)
    if by1 == 1 and by2 == ny:
        return ccd, oscan, 1
    cropped = ccdp.trim_image(ccd.copy(), fits_section=format_fits_section(1, nx, by1, by2))
    return cropped, oscan, by1


def _shift_section_y(sec: str, first_fits_row: int) -> str:
    """Rewrite a FITS section onto an image whose row 1 was ``first_fits_row``."""
    from .sanity import format_fits_section, section_bounds

    x1, x2, y1, y2 = section_bounds(sec)
    shift = int(first_fits_row) - 1
    return format_fits_section(x1, x2, y1 - shift, y2 - shift)


def apply_overscan_and_trim(
    ccd: CCDData,
    cfg: dict,
    *,
    verbose: bool = False,
    label: str = "",
) -> CCDData:
    """Subtract overscan (``BIASSEC``), then trim to ``DATASEC``.

    Methodology
    -----------
    Overscan columns measure the per-frame electronic pedestal. Subtracting
    them puts every frame on a common zero before bias/flat work.

    ccdproc first collapses the overscan to one median (or mean) per row.
    With ``overscan.model`` set (default ``legendre``, degree
    ``overscan.order``) that noisy row vector is fitted with a low-order
    polynomial and the *fit* is subtracted — what IRAF ``ccdproc``/``colbias``
    do with ``function=legendre``. With ``model: none`` the raw per-row values
    are subtracted; on a narrow overscan each row's error (read noise / √width)
    is stamped across the image as horizontal stripes that differ frame to frame.

    **Bias frames must be overscan-calibrated before combining** so the master
    bias holds residual 2-D structure relative to overscan — not an absolute
    ADU pedestal that would mismatch science frames that got their *own*
    overscan removed. See GUIDE['bias_calibrate'] / GUIDE['overscan'].

    Implementation: ``ccdproc.subtract_overscan`` + ``ccdproc.trim_image``
    (https://ccdproc.readthedocs.io/).
    """
    biassec = resolve_biassec(ccd.meta, cfg)
    datasec = resolve_datasec(ccd.meta, cfg)
    from .sanity import clip_section_y

    # If the usable overscan starts at y>1, those rows must leave the product.
    datasec = clip_section_y(datasec, biassec)

    from .term import dim

    prefix = f"    [{label}] " if label else "    "
    shape0 = ccd.shape
    settings = overscan_settings(cfg)
    estimate = describe_overscan_estimate(cfg)
    # Fit only the rows inside BIASSEC. ccdproc fits every row of the array,
    # so rows outside that section are removed before the call.
    work, oscan_sec, first_row = _crop_to_biassec_rows(ccd, biassec)
    trim_on_work = _shift_section_y(datasec, first_row)

    if verbose:
        print(
            dim(
                f"{prefix}overscan subtract: section={biassec}  "
                f"estimate={estimate}  (see {GUIDE['overscan']})"
            )
        )
    # Side overscan → per-row estimate (ccdproc overscan_axis=1 default),
    # optionally smoothed by a fitted polynomial along those rows only.
    reduced = ccdp.subtract_overscan(
        work,
        fits_section=oscan_sec,
        median=settings["median"],
        model=overscan_model(cfg),
    )
    if verbose:
        print(dim(f"{prefix}trim: section={datasec}  shape {shape0} → "), end="")
    reduced = ccdp.trim_image(reduced, fits_section=trim_on_work)
    if verbose:
        print(dim(f"{reduced.shape}"))

    _append_history(reduced, f"ccd_pipeline: overscan subtract {biassec} ({estimate})")
    _append_history(reduced, f"ccd_pipeline: trim to {datasec}")
    return reduced


def subtract_bias(
    ccd: CCDData,
    master_bias: CCDData,
    *,
    verbose: bool = False,
    label: str = "",
) -> CCDData:
    """Subtract the combined master bias (``ccdproc.subtract_bias``).

    Removes residual 2-D bias structure after overscan. Master bias should
    already be overscan-trimmed (GUIDE['bias_combine']).
    """
    from .term import dim

    prefix = f"    [{label}] " if label else "    "
    if verbose:
        print(dim(f"{prefix}subtract master bias"))
    out = ccdp.subtract_bias(ccd, master_bias)
    _append_history(out, "ccd_pipeline: subtract master bias")
    return out


def subtract_dark_current(
    ccd: CCDData,
    master_dark: CCDData,
    *,
    exptime_key: str = "EXPTIME",
    verbose: bool = False,
    label: str = "",
) -> CCDData:
    """Subtract a master dark, scaled by exposure time when both are known.

    Guide practice: dark frames are overscan- and bias-calibrated first, then
    combined. ``ccdproc.subtract_dark(..., scale=True)`` multiplies that master
    by ``(frame exposure) / (dark exposure)`` so a 30 s flat does not receive
    a 300 s dark. Without an exposure time the master is subtracted as-is.
    """
    import astropy.units as u

    from .term import dim

    data_exp = exposure_seconds(ccd.meta, exptime_key)
    dark_exp = exposure_seconds(master_dark.meta, exptime_key)
    prefix = f"    [{label}] " if label else "    "
    scale = data_exp is not None and dark_exp is not None
    if verbose:
        if scale:
            print(dim(f"{prefix}subtract master dark scaled {data_exp:g}s/{dark_exp:g}s"))
        else:
            print(dim(f"{prefix}subtract master dark (no exposure scale)"))
    if scale:
        out = ccdp.subtract_dark(
            ccd,
            master_dark,
            dark_exposure=dark_exp * u.s,
            data_exposure=data_exp * u.s,
            exposure_unit=u.s,
            scale=True,
        )
        _append_history(
            out,
            f"ccd_pipeline: subtract master dark scaled {data_exp:g}/{dark_exp:g}",
        )
    else:
        # ccdproc requires exposure arguments even when scale is off.
        out = ccdp.subtract_dark(
            ccd,
            master_dark,
            dark_exposure=1.0 * u.s,
            data_exposure=1.0 * u.s,
            scale=False,
        )
        _append_history(out, "ccd_pipeline: subtract master dark unscaled")
    return out


def flat_correct(
    ccd: CCDData,
    master_flat: CCDData,
    *,
    verbose: bool = False,
    label: str = "",
) -> CCDData:
    """Divide by the master flat (``ccdproc.flat_correct``) — multiplicative.

    Corrects pixel-to-pixel QE and large-scale illumination (vignetting, dust).
    Does **not** remove fringing (additive airglow interference); that is a
    separate post-flat step in ``fringe.py``. See GUIDE['flat_about'].
    """
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
    """σ-clip combine via ``ccdproc.combine`` (guide image-combination practice).

    Optional ``scale`` (e.g. ``inv_median``) normalizes frames before the
    combine — required for flats (GUIDE['flat_combine']).
    Docs: https://ccdproc.readthedocs.io/en/stable/image_combination.html
    """
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

    if verbose:
        from .term import dim

        scale_txt = "inv_median" if scale is not None else "none"
        print(
            dim(
                f"    combine: method={method}  sigma_clip={sigma_clip} "
                f"({sigma_clip_low}/{sigma_clip_high})  scale={scale_txt}"
            )
        )

    combined = ccdp.combine(paths_or_ccds, **kwargs)
    combined.meta["combined"] = True
    return combined
