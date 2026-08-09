"""Plate-solve calibrated science frames with offline Astrometry.net.

Uses ``solve-field`` (astrometry.net) with HDI-style pointing hints
(``RASTRNG`` / ``DECSTRNG``) and a configured pixel scale. Intended as a
post-``reduce`` step so later stacking can align via WCS / ``reproject``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
import astropy.units as u

from .term import S, dim, info, style, success, warn


DEFAULT_WCS = {
    "enabled": True,
    "solver": "astrometry-net",
    "solve_field": "solve-field",
    "index_dir": None,
    "scale_arcsec_per_pix": 0.43,
    "scale_tol_frac": 0.05,
    "radius_deg": 1.0,
    "downsample": 2,
    "overwrite": False,
    "timeout_sec": 120,
}


def wcs_config(cfg: dict) -> dict[str, Any]:
    """Merge night ``wcs:`` block with HDI-oriented defaults."""
    out = dict(DEFAULT_WCS)
    out.update(cfg.get("wcs") or {})
    return out


def parse_pointing(
    header,
    *,
    ra_keys: tuple[str, ...] = ("RASTRNG", "RA", "TELRA", "OBJCTRA"),
    dec_keys: tuple[str, ...] = ("DECSTRNG", "DEC", "TELDEC", "OBJCTDEC"),
) -> SkyCoord | None:
    """Return ICRS pointing from common header keywords, or ``None``."""
    ra_raw = None
    dec_raw = None
    for key in ra_keys:
        if key in header and header[key] not in (None, ""):
            ra_raw = header[key]
            break
    for key in dec_keys:
        if key in header and header[key] not in (None, ""):
            dec_raw = header[key]
            break
    if ra_raw is None or dec_raw is None:
        return None

    try:
        # Sexagesimal strings (HDI RASTRNG/DECSTRNG) or numeric degrees
        if isinstance(ra_raw, (int, float)) and isinstance(dec_raw, (int, float)):
            return SkyCoord(float(ra_raw) * u.deg, float(dec_raw) * u.deg, frame="icrs")
        ra = Angle(str(ra_raw).strip(), unit=u.hourangle)
        dec = Angle(str(dec_raw).strip(), unit=u.deg)
        return SkyCoord(ra=ra, dec=dec, frame="icrs")
    except Exception:
        return None


def has_wcs(header) -> bool:
    """True if a usable celestial WCS appears to be present."""
    ctype1 = str(header.get("CTYPE1", "")).upper()
    return ctype1.startswith("RA") or ctype1.startswith("GLON")


def build_solve_field_cmd(
    image_path: Path,
    *,
    out_fits: Path,
    work_dir: Path,
    wcs_cfg: dict[str, Any],
    pointing: SkyCoord | None,
) -> list[str]:
    """Construct ``solve-field`` argv (does not run it)."""
    scale = float(wcs_cfg["scale_arcsec_per_pix"])
    tol = float(wcs_cfg["scale_tol_frac"])
    scale_lo = scale * (1.0 - tol)
    scale_hi = scale * (1.0 + tol)
    solve_field = str(wcs_cfg.get("solve_field") or "solve-field")

    cmd: list[str] = [
        solve_field,
        "--dir",
        str(work_dir),
        "--no-plots",
        "--no-verify",
        "--overwrite",
        # Avoid optional Python helper steps that break on newer NumPy (np.bool).
        "--no-remove-lines",
        "--uniformize",
        "0",
        "--new-fits",
        str(out_fits),
        "--scale-units",
        "arcsecperpix",
        "--scale-low",
        f"{scale_lo:.6g}",
        "--scale-high",
        f"{scale_hi:.6g}",
    ]

    downsample = wcs_cfg.get("downsample")
    if downsample:
        cmd.extend(["--downsample", str(int(downsample))])

    index_dir = wcs_cfg.get("index_dir")
    if index_dir:
        cmd.extend(["--index-dir", str(index_dir)])

    if pointing is not None:
        cmd.extend(
            [
                "--ra",
                f"{pointing.ra.degree:.6f}",
                "--dec",
                f"{pointing.dec.degree:.6f}",
                "--radius",
                f"{float(wcs_cfg['radius_deg']):.4g}",
            ]
        )

    cmd.append(str(image_path))
    return cmd


def _append_history(header: fits.Header, line: str) -> None:
    if hasattr(header, "add_history"):
        header.add_history(line)
    else:
        header["HISTORY"] = line


def _merge_wcs_into_science(science_path: Path, solved_path: Path) -> None:
    """Copy WCS (and related) cards from solved FITS onto the science product."""
    with fits.open(solved_path, memmap=False) as solved:
        solved_hdr = solved[0].header.copy()

    with fits.open(science_path, mode="update", memmap=False) as hdul:
        hdr = hdul[0].header
        # Remove prior WCS-ish cards so we don't leave a mixed solution.
        # Do not touch observational EQUINOX (HDI decimal-year epoch).
        for key in list(hdr.keys()):
            ku = str(key).upper()
            if ku.startswith(
                (
                    "CD1_",
                    "CD2_",
                    "PC1_",
                    "PC2_",
                    "PV1_",
                    "PV2_",
                    "CRPIX",
                    "CRVAL",
                    "CTYPE",
                    "CUNIT",
                    "CDELT",
                    "CROTA",
                    "LONPOLE",
                    "LATPOLE",
                    "RADESYS",
                    "WCSAXES",
                    "WCSNAME",
                    "A_",
                    "B_",
                    "AP_",
                    "BP_",
                )
            ) or ku in {"RADECSYS", "IMAGEW", "IMAGEH"}:
                try:
                    del hdr[key]
                except KeyError:
                    pass

        for card in solved_hdr.cards:
            key = card.keyword
            ku = key.upper()
            if ku.startswith(
                (
                    "CD1_",
                    "CD2_",
                    "PC1_",
                    "PC2_",
                    "PV1_",
                    "PV2_",
                    "CRPIX",
                    "CRVAL",
                    "CTYPE",
                    "CUNIT",
                    "CDELT",
                    "CROTA",
                    "LONPOLE",
                    "LATPOLE",
                    "RADESYS",
                    "WCSAXES",
                    "WCSNAME",
                    "A_",
                    "B_",
                    "AP_",
                    "BP_",
                )
            ) or ku in {"RADECSYS", "IMAGEW", "IMAGEH"}:
                hdr[key] = card.value

        for key in ("SOLVED", "PARITY", "ASTIRMS", "ASTRRMS"):
            if key in solved_hdr:
                hdr[key] = solved_hdr[key]

        _append_history(hdr, "ccd_pipeline: WCS via solve-field (astrometry.net)")
        hdul.flush()


def solve_one_frame(
    path: Path,
    *,
    wcs_cfg: dict[str, Any],
    overwrite: bool,
) -> tuple[str, str]:
    """
    Plate-solve one science FITS in place.

    Returns ``(status, message)`` where status is
    ``ok`` / ``skip`` / ``fail``.
    """
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        if has_wcs(header) and not overwrite:
            return "skip", "already has WCS (pass --overwrite to re-solve)"
        pointing = parse_pointing(header)

    solve_bin = str(wcs_cfg.get("solve_field") or "solve-field")
    if shutil.which(solve_bin) is None and not Path(solve_bin).is_file():
        return "fail", f"solve-field not found: {solve_bin!r} (install astrometry.net)"

    with tempfile.TemporaryDirectory(prefix="ccd_wcs_") as tmp:
        work = Path(tmp)
        # Work on a copy so solve-field sidecar junk stays in tmp
        work_in = work / path.name
        shutil.copy2(path, work_in)
        out_fits = work / f"{path.stem}.wcs.fits"
        cmd = build_solve_field_cmd(
            work_in,
            out_fits=out_fits,
            work_dir=work,
            wcs_cfg=wcs_cfg,
            pointing=pointing,
        )
        timeout = float(wcs_cfg.get("timeout_sec") or 120)
        # System solve-field shells out to /usr/bin/python3. A newer user-site
        # NumPy (no np.bool) breaks removelines/uniformize — isolate that.
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return "fail", f"solve-field timed out after {timeout:.0f}s"
        except FileNotFoundError:
            return "fail", f"solve-field not found: {solve_bin!r}"

        if proc.returncode != 0 or not out_fits.exists():
            # Also accept default .new name if --new-fits was ignored
            alt = work / f"{work_in.stem}.new"
            if alt.exists():
                out_fits = alt
            else:
                blob = (proc.stderr or "") + "\n" + (proc.stdout or "")
                if "np.bool" in blob or "numpy" in blob.lower() and "bool" in blob:
                    return (
                        "fail",
                        "solve-field Python helpers crashed (NumPy incompatibility). "
                        "Pipeline sets PYTHONNOUSERSITE=1; if this persists, upgrade "
                        "astrometry.net or remove ~/.local NumPy for system python3.",
                    )
                tail = blob.strip().splitlines()
                detail = tail[-1] if tail else f"exit {proc.returncode}"
                hint = ""
                if pointing is None:
                    hint = " (no RASTRNG/DEC pointing in header)"
                return "fail", f"no solution — {detail}{hint}"

        _merge_wcs_into_science(path, out_fits)

    coord = ""
    if pointing is not None:
        coord = (
            f"  hint RA={pointing.ra.to_string(unit=u.hour, sep=':', precision=1)} "
            f"Dec={pointing.dec.to_string(unit=u.deg, sep=':', precision=0)}"
        )
    return "ok", f"WCS written{coord}"


def iter_science_fits(cfg: dict) -> list[Path]:
    sci_dir = Path(cfg["paths"]["output_dir"]) / "science"
    if not sci_dir.is_dir():
        return []
    return sorted(p for p in sci_dir.glob("*.fits") if p.is_file())


def _wcs_log_path(cfg: dict) -> Path:
    """``<output_dir>/wcs_solve.log`` (night reduced folder)."""
    return Path(cfg["paths"]["output_dir"]) / "wcs_solve.log"


def _open_wcs_log(cfg: dict) -> tuple[Path, TextIO]:
    log_path = _wcs_log_path(cfg)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path, log_path.open("a", encoding="utf-8")


def _log_line(fh: TextIO | None, line: str) -> None:
    if fh is None:
        return
    fh.write(line.rstrip() + "\n")
    fh.flush()


def solve_science_wcs(
    cfg: dict,
    *,
    limit: int | None = None,
    overwrite: bool | None = None,
) -> dict[str, Any]:
    """Plate-solve calibrated science frames under ``<output_dir>/science/``."""
    wcfg = wcs_config(cfg)
    if overwrite is not None:
        wcfg["overwrite"] = bool(overwrite)

    if not wcfg.get("enabled", True):
        warn("wcs.enabled is false in config — nothing to do")
        return {"ok": 0, "skip": 0, "fail": 0, "paths": [], "log": None}

    if str(wcfg.get("solver", "astrometry-net")) not in {"astrometry-net", "solve-field"}:
        raise RuntimeError(
            f"Unsupported wcs.solver={wcfg.get('solver')!r}; only astrometry-net is implemented"
        )

    paths = iter_science_fits(cfg)
    if limit is not None:
        paths = paths[:limit]

    out_dir = Path(cfg["paths"]["output_dir"])
    sci_dir = out_dir / "science"
    info(f"Science directory: {sci_dir}")
    info(f"Solver           : {wcfg.get('solve_field', 'solve-field')}")
    info(
        f"Scale            : {wcfg['scale_arcsec_per_pix']}″/pix "
        f"(±{100*float(wcfg['scale_tol_frac']):.1f}%)"
    )
    print(f"Frames to solve: {len(paths)}")

    log_path: Path | None = None
    log_fh: TextIO | None = None
    try:
        log_path, log_fh = _open_wcs_log(cfg)
        info(f"Log              : {log_path}")
        started = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        _log_line(log_fh, "")
        _log_line(log_fh, "=" * 72)
        _log_line(log_fh, f"WCS solve run  {started}")
        _log_line(log_fh, f"night_id       {cfg.get('night_id', '')}")
        _log_line(log_fh, f"science_dir    {sci_dir}")
        _log_line(log_fh, f"solver         {wcfg.get('solve_field', 'solve-field')}")
        _log_line(
            log_fh,
            f"scale_arcsec   {wcfg['scale_arcsec_per_pix']} "
            f"(±{100*float(wcfg['scale_tol_frac']):.1f}%)",
        )
        _log_line(log_fh, f"overwrite      {bool(wcfg.get('overwrite'))}")
        _log_line(log_fh, f"limit          {limit}")
        _log_line(log_fh, f"n_frames       {len(paths)}")
        _log_line(log_fh, "-" * 72)

        if not paths:
            warn("No science FITS found — run ccd reduce first")
            _log_line(log_fh, "RESULT: no science FITS found")
            return {"ok": 0, "skip": 0, "fail": 0, "paths": [], "log": str(log_path)}

        counts = {"ok": 0, "skip": 0, "fail": 0}
        solved_paths: list[str] = []

        for i, path in enumerate(paths, start=1):
            print()
            print(style(f"[{i}/{len(paths)}] {path.name}", S.BOLD, S.MAGENTA))
            status, msg = solve_one_frame(
                path, wcs_cfg=wcfg, overwrite=bool(wcfg.get("overwrite"))
            )
            counts[status] = counts.get(status, 0) + 1
            _log_line(log_fh, f"[{i}/{len(paths)}] {status.upper():<4}  {path.name}  {msg}")
            if status == "ok":
                print(dim(f"    {msg}"))
                solved_paths.append(str(path))
            elif status == "skip":
                warn(f"    SKIP: {msg}")
            else:
                warn(f"    FAIL: {msg}")

        print()
        summary = (
            f"WCS solve: {counts['ok']} ok, {counts['skip']} skipped, {counts['fail']} failed"
        )
        _log_line(log_fh, "-" * 72)
        _log_line(log_fh, summary)
        _log_line(log_fh, f"log file: {log_path}")
        if counts["fail"] and not counts["ok"]:
            warn(summary)
        else:
            success(summary)
        info(f"Wrote log: {log_path}")
        return {**counts, "paths": solved_paths, "log": str(log_path)}
    finally:
        if log_fh is not None:
            log_fh.close()
