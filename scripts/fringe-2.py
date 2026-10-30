#!/usr/bin/env python
"""fringe-2.py — build a fringe frame the IRAF / NOAO way, as a cross-check.

This is a deliberately *independent* implementation of the canonical NOAO
recipe for making a fringe correction image from flat-fielded sky exposures
that still contain stars.  It follows the two IRAF tasks that defined the
method and are still the most widely cited reference for it:

  1. ``imcombine`` (images.immatch)  — stack the sky frames.
       * ``scale = median``  : every frame is brought to a common sky level
       * ``combine = median``: per-pixel median of the scaled frames
       * ``reject = ccdclip``: iterative sigma clipping with a CCD noise
         model  σ² = (rdnoise/gain)² + value/gain  — symmetric low/high
         clipping about the median, so (unlike minmax / high-only rejection)
         it does not bias the result where fewer frames survive.
       https://iraf.readthedocs.io/en/latest/tasks/images/immatch/imcombine.html

  2. ``mkfringecor`` (noao.imred.ccdred) — turn the combined sky into a
     fringe image.
       * The slowly varying background is a moving *boxcar* average whose
         box grows from ``xboxmin``/``yboxmin`` (5 px) at the image edges to
         ``xboxmax``/``yboxmax`` (0.25 × image size) in the middle.
       * Objects **and the fringes themselves** are excluded from that
         average by sigma clipping (``lowsigma = highsigma = 2.5``) relative
         to the smoothed background, using the rms of the image lines.
       * fringe = sky − background   (additive, in ADU)
       * The doc is explicit that "objects left in the fringe image will
         affect the fringe corrected observations" — nothing is interpolated
         or painted over.  Whatever the combine did not reject stays.
       https://iraf.readthedocs.io/en/latest/tasks/noao/imred/ccdred/mkfringecor.html

  Apply (``ccdproc fringecor`` / ``rmfringe``): science − (sky_sci/FRGSKY) × fringe.

Nothing here imports from ``ccd_pipeline`` so it is a genuinely separate
test.  Inputs must already be overscan/bias/flat corrected.

Usage
-----
    python scripts/fringe-2.py --out masters/master_fringe2_z.fits \
        science/2016OCT11.Blank_Sky.z.0[124].fits \
        --gain 1.3 --compare masters/master_fringe_z.fits \
        --preview masters/diagnostics/fringe2_vs_pipeline_z.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import uniform_filter1d


# --------------------------------------------------------------------------
# Step 1 — imcombine: scale=median, combine=median, reject=ccdclip
# --------------------------------------------------------------------------
def robust_median(a: np.ndarray) -> float:
    """Median of finite pixels (imcombine 'median' scaling statistic)."""
    return float(np.nanmedian(a))


def imcombine_ccdclip(
    frames: list[np.ndarray],
    *,
    gain: float,
    rdnoise: float,
    lsigma: float = 3.0,
    hsigma: float = 3.0,
    niter: int = 3,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    """Return (combined, nkept, reference_sky, scales).

    * scales[i]  multiplies frame i so its median equals the reference sky
      (the mean of the input medians, as imcombine does).
    * combined   per-pixel median of the surviving scaled samples.
    * nkept      number of samples that survived clipping at each pixel.
    """
    meds = [robust_median(f) for f in frames]
    ref = float(np.mean(meds))
    scales = [ref / m for m in meds]

    stack = np.stack([f * s for f, s in zip(frames, scales)], axis=0)
    # CCD noise model per *scaled* sample: σ = s·sqrt((rn/g)² + max(v,0)/g)
    var = np.stack(
        [
            (s**2) * ((rdnoise / gain) ** 2 + np.clip(f, 0, None) / gain)
            for f, s in zip(frames, scales)
        ],
        axis=0,
    )
    sig = np.sqrt(var)
    del var

    good = np.isfinite(stack)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(niter):
            work = np.where(good, stack, np.nan)
            center = np.nanmedian(work, axis=0)  # mclip=yes: median centre
            dev = stack - center[np.newaxis]
            keep = good & (dev >= -lsigma * sig) & (dev <= hsigma * sig)
            # imcombine never rejects the last sample at a pixel
            n_keep = keep.sum(axis=0)
            keep |= good & (n_keep[np.newaxis] == 0) & np.isfinite(stack)
            if np.array_equal(keep, good):
                break
            good = keep
        work = np.where(good, stack, np.nan)
        combined = np.nanmedian(work, axis=0)
    nkept = good.sum(axis=0).astype(np.int16)
    return combined, nkept, ref, scales


# --------------------------------------------------------------------------
# Step 2 — mkfringecor: variable-box boxcar background with sigma clipping
# --------------------------------------------------------------------------
def _box_sizes(n: int, boxmin: int, boxmax: float) -> np.ndarray:
    """Per-index box length along one axis: boxmin at the edges, growing to
    boxmax at the centre (IRAF xboxmin/xboxmax semantics)."""
    bmax = int(round(boxmax * n)) if boxmax < 1 else int(boxmax)
    bmax = max(boxmin, min(bmax, n))
    idx = np.arange(n)
    dist = np.minimum(idx, n - 1 - idx)  # distance to nearest edge
    box = np.clip(2 * dist + 1, boxmin, bmax)
    return box.astype(int)


def _variable_boxcar_1d(arr: np.ndarray, axis: int, boxes: np.ndarray) -> np.ndarray:
    """Moving average along ``axis`` where the box length varies with position.

    Implemented as a ladder of fixed-size ``uniform_filter1d`` passes; each
    position takes the result of the largest ladder size ≤ its own box.
    ``mode='constant'`` (zero padding) is fine because the caller divides a
    data·weight sum by a weight sum computed the same way.
    """
    ladder = sorted({int(b) for b in np.geomspace(boxes.min(), boxes.max(), 12).round()})
    ladder = sorted({max(1, b) for b in ladder} | {int(boxes.min()), int(boxes.max())})
    out = np.empty_like(arr)
    # map each position to its ladder size
    choice = np.zeros_like(boxes)
    for b in ladder:
        choice[boxes >= b] = b
    for b in ladder:
        sel = choice == b
        if not sel.any():
            continue
        filt = uniform_filter1d(arr, size=b, axis=axis, mode="constant", cval=0.0)
        if axis == 0:
            out[sel, :] = filt[sel, :]
        else:
            out[:, sel] = filt[:, sel]
    return out


def variable_boxcar_mean(
    data: np.ndarray,
    weight: np.ndarray,
    *,
    boxmin: int,
    boxmax: float,
) -> np.ndarray:
    """Weighted moving average with the IRAF edge-shrinking box (separable)."""
    ny, nx = data.shape
    ybox = _box_sizes(ny, boxmin, boxmax)
    xbox = _box_sizes(nx, boxmin, boxmax)
    num = np.where(weight > 0, data * weight, 0.0)
    den = weight.astype(float)
    # Sums are normalised by the same (count-preserving) filtering so the
    # result equals sum(data·w)/sum(w) over the box.
    for arr_name in ("num", "den"):
        arr = num if arr_name == "num" else den
        arr = _variable_boxcar_1d(arr, 1, xbox)
        arr = _variable_boxcar_1d(arr, 0, ybox)
        if arr_name == "num":
            num = arr
        else:
            den = arr
    with np.errstate(invalid="ignore", divide="ignore"):
        bkg = num / den
    return bkg


def mkfringecor(
    sky: np.ndarray,
    *,
    boxmin: int = 5,
    boxmax: float = 0.25,
    lowsigma: float = 2.5,
    highsigma: float = 2.5,
    niter: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fringe, background, clipmask) following IRAF mkfringecor.

    The background is a heavily smoothed boxcar average; objects *and fringes*
    are excluded from it by clipping at ±(low/high)σ relative to the current
    background, with σ measured per image line (as the IRAF doc describes).
    """
    finite = np.isfinite(sky)
    work = np.where(finite, sky, 0.0)
    weight = finite.astype(float)
    bkg = variable_boxcar_mean(work, weight, boxmin=boxmin, boxmax=boxmax)
    clip = ~finite
    for _ in range(niter):
        resid = np.where(finite, work - bkg, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # rms of each image line relative to the smoothed background
            line_rms = np.nanstd(np.where(clip, np.nan, resid), axis=1, keepdims=True)
        line_rms = np.where(np.isfinite(line_rms) & (line_rms > 0), line_rms, np.nanmedian(line_rms))
        new_clip = ~finite | (resid < -lowsigma * line_rms) | (resid > highsigma * line_rms)
        if np.array_equal(new_clip, clip):
            break
        clip = new_clip
        weight = (~clip).astype(float)
        bkg = variable_boxcar_mean(work, weight, boxmin=boxmin, boxmax=boxmax)
    fringe = np.where(finite, sky - bkg, 0.0)
    return fringe, bkg, clip


# --------------------------------------------------------------------------
# Optional comparison with the pipeline master (fraction-of-sky template)
# --------------------------------------------------------------------------
def compare_and_preview(
    fringe_adu: np.ndarray,
    ref_sky: float,
    nkept: np.ndarray,
    compare_path: Path | None,
    preview_path: Path | None,
    *,
    filt: str,
) -> None:
    f2 = fringe_adu / ref_sky  # fraction of sky, like master_fringe_<filt>.fits
    from scipy.ndimage import gaussian_filter

    def split(a):
        lo = gaussian_filter(np.nan_to_num(a), 3.0)
        return lo, a - lo

    lo2, hi2 = split(f2)
    print(f"fringe-2 : rms={np.nanstd(f2):.4f}  low-freq(σ=3) rms={np.std(lo2):.4f}  "
          f"per-pixel noise rms={np.nanstd(hi2):.4f}")

    f1 = None
    if compare_path is not None and Path(compare_path).exists():
        f1 = np.asarray(fits.getdata(compare_path), dtype=float)
        if f1.shape == f2.shape:
            lo1, hi1 = split(f1)
            d = f2 - f1
            ok = np.isfinite(d) & (f1 != 0)
            cc_lo = np.corrcoef(lo1[ok].ravel(), lo2[ok].ravel())[0, 1]
            cc_all = np.corrcoef(f1[ok].ravel(), f2[ok].ravel())[0, 1]
            print(f"pipeline : rms={np.nanstd(f1):.4f}  low-freq(σ=3) rms={np.std(lo1):.4f}  "
                  f"per-pixel noise rms={np.nanstd(hi1):.4f}")
            print(f"difference (fringe-2 − pipeline): rms={np.std(d[ok]):.4f}  "
                  f"median={np.median(d[ok]):+.5f}")
            print(f"correlation: all pixels r={cc_all:.3f}   low-frequency (σ=3) r={cc_lo:.3f}")
        else:
            print(f"compare: shape mismatch {f1.shape} vs {f2.shape}; skipping")
            f1 = None

    if preview_path is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = max(1, max(f2.shape) // 1024)
    panels = [(f"fringe-2 (IRAF imcombine+mkfringecor) '{filt}'", f2)]
    if f1 is not None:
        panels.insert(0, (f"pipeline master_fringe_{filt}", f1))
        panels.append(("fringe-2 − pipeline", f2 - f1))
    panels.append(("samples kept per pixel (ccdclip)", nkept.astype(float)))
    lim = float(np.nanpercentile(np.abs(f2[::step, ::step]), 99)) or 1e-3
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.2), squeeze=False)
    for ax, (title, img) in zip(axes[0], panels):
        sub = img[::step, ::step]
        if "samples" in title:
            im = ax.imshow(sub, origin="lower", cmap="viridis", vmin=0, vmax=float(sub.max()))
            fig.colorbar(im, ax=ax, fraction=0.046, label="frames")
        else:
            im = ax.imshow(sub, origin="lower", cmap="gray", vmin=-lim, vmax=lim)
            fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of sky")
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_path, dpi=120)
    plt.close(fig)
    print(f"preview → {preview_path}")


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="flat-fielded sky frames (same filter)")
    ap.add_argument("--out", type=Path, required=True, help="output fringe FITS (ADU, additive)")
    ap.add_argument("--gain", type=float, default=1.3, help="e-/ADU for ccdclip noise model (HDI ≈ 1.3)")
    ap.add_argument("--rdnoise", type=float, default=None, help="e- read noise; default from RDNOISE card or 5.0")
    ap.add_argument("--lsigma", type=float, default=3.0, help="imcombine ccdclip low sigma")
    ap.add_argument("--hsigma", type=float, default=3.0, help="imcombine ccdclip high sigma")
    ap.add_argument("--boxmin", type=int, default=5, help="mkfringecor x/yboxmin (px)")
    ap.add_argument("--boxmax", type=float, default=0.25, help="mkfringecor x/yboxmax (<1 = fraction of image)")
    ap.add_argument("--lowsigma", type=float, default=2.5, help="mkfringecor lowsigma")
    ap.add_argument("--highsigma", type=float, default=2.5, help="mkfringecor highsigma")
    ap.add_argument("--compare", type=Path, default=None, help="pipeline master_fringe_<f>.fits to compare against")
    ap.add_argument("--preview", type=Path, default=None, help="PNG side-by-side preview")
    args = ap.parse_args(argv)

    inputs = [p for p in args.inputs if not p.name.startswith(".")]
    if len(inputs) < 2:
        print("need ≥2 input frames", file=sys.stderr)
        return 2

    frames, hdrs = [], []
    for p in inputs:
        with fits.open(p, memmap=False) as hdul:
            frames.append(np.asarray(hdul[0].data, dtype=float))
            hdrs.append(hdul[0].header)
        print(f"  → {p.name}  median={robust_median(frames[-1]):.4g} ADU")
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        print(f"shape mismatch among inputs: {shapes}", file=sys.stderr)
        return 2

    rdnoise = args.rdnoise
    if rdnoise is None:
        rdnoise = float(hdrs[0].get("RDNOISE", 5.0) or 5.0)
    filt = str(hdrs[0].get("FILTER", "?")).strip()

    print(f"[1] imcombine: scale=median combine=median reject=ccdclip "
          f"(gain={args.gain}, rdnoise={rdnoise}, lsig={args.lsigma}, hsig={args.hsigma})")
    combined, nkept, ref_sky, scales = imcombine_ccdclip(
        frames, gain=args.gain, rdnoise=rdnoise, lsigma=args.lsigma, hsigma=args.hsigma
    )
    for p, s in zip(inputs, scales):
        print(f"      {p.name}: scale ×{s:.4f}")
    n = len(frames)
    for k in range(n + 1):
        frac = float(np.mean(nkept == k))
        if frac:
            print(f"      pixels with {k}/{n} samples kept: {100*frac:6.2f}%")
    del frames

    print(f"[2] mkfringecor: boxmin={args.boxmin} boxmax={args.boxmax} "
          f"lowsigma={args.lowsigma} highsigma={args.highsigma}")
    fringe, bkg, clip = mkfringecor(
        combined,
        boxmin=args.boxmin,
        boxmax=args.boxmax,
        lowsigma=args.lowsigma,
        highsigma=args.highsigma,
    )
    print(f"      clipped from background estimate: {100*clip.mean():.2f}% of pixels")
    print(f"      fringe rms = {np.nanstd(fringe):.4g} ADU  = {np.nanstd(fringe)/ref_sky:.4f} of sky ({ref_sky:.4g} ADU)")

    hdr = fits.Header()
    hdr["IMAGETYP"] = "FRINGE"
    hdr["FILTER"] = filt
    hdr["BUNIT"] = ("adu", "additive fringe at FRGSKY sky level")
    hdr["FRGSKY"] = (ref_sky, "reference sky (ADU); apply science-(sky/FRGSKY)*fringe")
    hdr["NCOMBINE"] = n
    hdr["GAIN"] = (args.gain, "e-/ADU used in ccdclip")
    hdr["RDNOISE"] = (rdnoise, "e- used in ccdclip")
    hdr["CCDCLIP"] = (f"{args.lsigma},{args.hsigma}", "imcombine lsigma,hsigma")
    hdr["FRGBOX"] = (f"{args.boxmin},{args.boxmax}", "mkfringecor boxmin,boxmax")
    hdr["FRGCLIP"] = (f"{args.lowsigma},{args.highsigma}", "mkfringecor lowsigma,highsigma")
    for i, (p, s) in enumerate(zip(inputs, scales), 1):
        hdr[f"IMCMB{i:03d}"] = p.name
        hdr[f"SCALE{i:03d}"] = round(s, 6)
    hdr.add_history("fringe-2.py: imcombine(scale=median,combine=median,reject=ccdclip) "
                    "then mkfringecor(variable boxcar, 2.5-sigma clip); IRAF/NOAO recipe")
    hdr.add_history("https://iraf.readthedocs.io/en/latest/tasks/noao/imred/ccdred/mkfringecor.html")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(
        [
            fits.PrimaryHDU(data=fringe.astype(np.float32), header=hdr),
            fits.ImageHDU(data=nkept, name="NKEPT"),
            fits.ImageHDU(data=bkg.astype(np.float32), name="BKG"),
        ]
    ).writeto(args.out, overwrite=True)
    print(f"wrote {args.out}")

    compare_and_preview(fringe, ref_sky, nkept, args.compare, args.preview, filt=filt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
