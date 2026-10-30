"""Plate-solve calibrated science frames with offline Astrometry.net.

Uses ``solve-field`` (astrometry.net) with HDI-style pointing hints
(``RASTRNG`` / ``DECSTRNG``) and a configured pixel scale. Intended as a
post-``reduce`` / post-fringe step so later stacking can align via WCS /
``reproject`` (README Methodology §7).

Not part of photometric calibration (bias/flat/fringe). WCS enables geometric
alignment for combining; Astropy ``reproject`` and guide combining chapters
cover the stacking side:
https://reproject.readthedocs.io/
https://docs.astropy.org/en/stable/wcs/

Astrometry.net: https://astrometry.net/
``solve-field`` docs: https://astrometry.net/doc/readme.html
"""

from __future__ import annotations

import os
import re
import select
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
import astropy.units as u

from .io import iter_fits_files
from .reduce import remove_unprefixed_science_twin, retitle_science_file, split_science_stage_name
from .term import S, dim, info, show_ccd, show_command, style, success, warn


# Unbinned plate scales (arcsec/pixel). Matched as substrings of
# detector / instrument / telescope (lowercase).
# S2KB: 0.60″/pix, 2048×2048 → 20.5′ (https://www.wiyn.org/09m_web/observe/s2kb.html)
# HDI:  0.43″/pix, ~29′
# Hall 42-inch E2V CCD-231: header PIXSCAL=0.327 is stale. solve-field on
# 2026MAY16 2×2 frames measures 0.739″/pix (25′ field) → 0.369 unbinned.
_UNBINNED_SCALES = (
    ("s2kb", 0.60),
    ("hdi", 0.43),
    ("hall", 0.369),
    ("ccd-231", 0.369),
)


def bin_factor(ccdsum: str | None) -> float:
    """On-chip binning from a ``CCDSUM`` card such as ``2 2``."""
    if not ccdsum:
        return 1.0
    parts = str(ccdsum).replace(",", " ").split()
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return 1.0
    if not vals or vals[0] <= 0:
        return 1.0
    return vals[0]


def unbinned_pixscal(header) -> float | None:
    """Unbinned ″/pix from ``PIXSCAL`` / ``PIXSCALE`` (Hall: 0.327)."""
    if header is None:
        return None
    for key in ("PIXSCAL", "PIXSCALE"):
        raw = header.get(key)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def plate_scale_from_header(header) -> float | None:
    """Binned ″/pix from header ``PIXSCAL`` × ``CCDSUM``, or ``None``."""
    base = unbinned_pixscal(header)
    if base is None:
        return None
    return base * bin_factor(header.get("CCDSUM") if header is not None else None)


def plate_scale_arcsec(
    *,
    detector: str = "",
    instrument: str = "",
    telescope: str = "",
    ccdsum: str | None = None,
    pixscal_unbinned: float | None = None,
    header=None,
) -> float:
    """Arcsec per pixel in the file that will be solved, including binning.

    Named detector/telescope first (Hall ``PIXSCAL`` 0.327 is stale — the
    file scale is 0.369″/pix unbinned). Then header ``PIXSCAL``, then HDI.
    """
    if header is not None and ccdsum is None:
        ccdsum = header.get("CCDSUM")
        if pixscal_unbinned in (None, ""):
            pixscal_unbinned = unbinned_pixscal(header)
    blob = f"{detector} {instrument} {telescope}".lower()
    base = None
    for name, scale in _UNBINNED_SCALES:
        if name in blob:
            base = scale
            break
    if base is None and pixscal_unbinned not in (None, ""):
        try:
            value = float(pixscal_unbinned)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            base = value
    if base is None:
        base = 0.43
    return base * bin_factor(ccdsum)


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
    "timeout_sec": 300,
    # Keep the terminal quiet; full solve-field text still goes to wcs_logs/.
    "verbose": False,  # True → pass --verbose to solve-field (huge logs)
    "show_solver_output": False,  # True → stream every solve-field line to terminal
    "include_objects": [],  # empty = all science frames; else only these OBJECT names
}


def wcs_config(cfg: dict) -> dict[str, Any]:
    """Merge night ``wcs:`` block with HDI-oriented defaults."""
    out = dict(DEFAULT_WCS)
    out.update(cfg.get("wcs") or {})
    return out


def apply_header_plate_scale(wcfg: dict[str, Any], header) -> float | None:
    """Widen the solve-field scale window so it contains header PIXSCAL×binning.

    Hall ``PIXSCAL`` can be stale (0.327 unbinned vs 0.369 measured). Replacing
    the config with that card *tightens* the window onto the wrong value.
    Expanding so both the YAML scale and PIXSCAL×bin sit inside the window
    still includes the real 0.739″/pix at 2×2. Returns the center scale, or
    ``None`` when the header has no PIXSCAL.
    """
    header_scale = plate_scale_from_header(header)
    if header_scale is None:
        return None
    cfg_scale = float(wcfg.get("scale_arcsec_per_pix") or 0)
    if cfg_scale <= 0:
        wcfg["scale_arcsec_per_pix"] = header_scale
        return header_scale
    lo = min(header_scale, cfg_scale)
    hi = max(header_scale, cfg_scale)
    if hi / lo - 1.0 <= float(wcfg.get("scale_tol_frac") or 0.05) * 2:
        return cfg_scale
    mid = 0.5 * (lo + hi)
    need = max(1.0 - lo / mid, hi / mid - 1.0) + 0.03
    wcfg["scale_arcsec_per_pix"] = mid
    wcfg["scale_tol_frac"] = max(float(wcfg.get("scale_tol_frac") or 0.05), need)
    return mid


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

    timeout = float(wcs_cfg.get("timeout_sec") or 300)
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
        "--cpulimit",
        str(max(1, int(timeout))),
        "--new-fits",
        str(out_fits),
        "--scale-units",
        "arcsecperpix",
        "--scale-low",
        f"{scale_lo:.6g}",
        "--scale-high",
        f"{scale_hi:.6g}",
    ]

    if wcs_cfg.get("verbose", False):
        cmd.append("--verbose")

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


def _combine_solver_streams(stdout: str | None, stderr: str | None) -> str:
    parts = []
    if stdout and stdout.strip():
        parts.append(stdout.rstrip())
    if stderr and stderr.strip():
        # solve-field often puts progress on stderr
        if not parts or stderr.strip() not in parts[0]:
            parts.append(stderr.rstrip())
    return "\n".join(parts).strip()


def _print_solver_output(text: str, *, max_lines: int | None = None) -> None:
    if not text:
        print(dim("    (no solve-field output captured)"))
        return
    lines = text.splitlines()
    if max_lines is not None and len(lines) > max_lines:
        omitted = len(lines) - max_lines
        lines = lines[-max_lines:]
        print(dim(f"    … ({omitted} earlier lines omitted; see log for full text)"))
    for line in lines:
        print(dim(f"    | {line}"))


def _solver_fail_summary(text: str) -> str:
    """One-line diagnosis from solve-field text (peak odds / last index)."""
    if not text:
        return ""
    odds_vals: list[float] = []
    for m in re.finditer(r"Best odds encountered:\s*([0-9.eE+-]+)", text):
        try:
            odds_vals.append(float(m.group(1)))
        except ValueError:
            pass
    # Also catch mid-search peaks that may exceed the per-index "Best odds" line
    for m in re.finditer(r"logodds\s+([0-9.eE+-]+)", text, flags=re.IGNORECASE):
        try:
            odds_vals.append(float(m.group(1)))
        except ValueError:
            pass
    last_index = None
    for m in re.finditer(r"did not solve \(index ([^,]+)", text):
        last_index = m.group(1).strip()
    bits: list[str] = []
    if odds_vals:
        peak = max(odds_vals)
        bits.append(f"best log-odds {peak:.2g} (need ~30)")
    if last_index:
        bits.append(f"last index {last_index}")
    return "; ".join(bits)


# Astrometry.net typically accepts a solve near log-odds ≈ 30 (ln 1e13-ish).
SOLVE_ODDS_TARGET = 30.0


def _short_index_name(path_or_name: str) -> str:
    name = Path(path_or_name.rstrip(".")).name
    for suffix in (".fits", ".littleendian"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.startswith("index-"):
        name = name[len("index-") :]
    return name


def _parse_solve_progress(line: str, state: dict[str, Any]) -> bool:
    """Update ``state`` from a solve-field log line. Return True if UI should refresh."""
    changed = False
    s = line.strip()
    m = re.match(r"startdepth\s+(\d+)", s, flags=re.IGNORECASE)
    if m:
        state["depth_lo"] = int(m.group(1))
        return True
    m = re.match(r"enddepth\s+(\d+)", s, flags=re.IGNORECASE)
    if m:
        state["depth_hi"] = int(m.group(1))
        return True
    m = re.match(r"Trying index\s+(\S+)", s)
    if m:
        state["index"] = _short_index_name(m.group(1))
        return True
    m = re.match(r"Objs:\s*(\d+)\s*,\s*(\d+)", s)
    if m:
        state["depth_lo"] = int(m.group(1))
        state["depth_hi"] = int(m.group(2))
        return True

    odds: float | None = None
    m = re.search(r"Best odds encountered:\s*([0-9.eE+-]+)", s, flags=re.IGNORECASE)
    if m:
        try:
            odds = float(m.group(1))
        except ValueError:
            odds = None
    else:
        m = re.search(r"logodds\s+([0-9.eE+-]+)", s, flags=re.IGNORECASE)
        if m:
            try:
                odds = float(m.group(1))
            except ValueError:
                odds = None
    if odds is not None and odds > float(state.get("peak_odds") or 0):
        state["peak_odds"] = odds
        changed = True

    m = re.search(r"solved with index\s+(\S+)", s, flags=re.IGNORECASE)
    if m:
        state["solved"] = True
        state["solved_index"] = _short_index_name(m.group(1))
        return True
    if re.search(r"Field\s+\d+\s+solved", s, flags=re.IGNORECASE):
        state["solved"] = True
        # Prefer an explicit index on this line; else the last tried index.
        m = re.search(r"index\s+(\S+)", s, flags=re.IGNORECASE)
        if m:
            state["solved_index"] = _short_index_name(m.group(1))
        elif state.get("index") and not state.get("solved_index"):
            state["solved_index"] = state["index"]
        changed = True
    return changed


def _solver_solved_index(text: str) -> str | None:
    """Return the short index name that solved the field, if reported."""
    if not text:
        return None
    # Prefer the explicit success line.
    matches = list(re.finditer(r"solved with index\s+(\S+)", text, flags=re.IGNORECASE))
    if matches:
        return _short_index_name(matches[-1].group(1))
    # Fallback: last "Trying index" before a "Field N solved" line.
    last_try: str | None = None
    for line in text.splitlines():
        m = re.match(r"Trying index\s+(\S+)", line.strip())
        if m:
            last_try = _short_index_name(m.group(1))
            continue
        if re.search(r"Field\s+\d+\s+solved|solved with index", line, flags=re.IGNORECASE):
            if last_try:
                return last_try
    return last_try


def _closeness_label(peak: float | None, *, target: float = SOLVE_ODDS_TARGET) -> str:
    """Human label for how close peak log-odds are to a typical solve."""
    if peak is None:
        return f"match — (need ~{target:.0f})"
    if peak >= target:
        # Values of 100–300 are normal after a real solve; not a percentage.
        return f"match {peak:.3g} (strong; need ~{target:.0f})"
    if peak <= 1.05:
        return f"match {peak:.3g}/{target:.0f} (far)"
    pct = min(99, int(round(100.0 * peak / target))) if target > 0 else 0
    return f"match {peak:.3g}/{target:.0f} ({pct}%)"


def _wcs_progress_line(elapsed: int, state: dict[str, Any], *, hint: str | None = None) -> str:
    """Short status line for the terminal (timer + solved/odds only)."""
    parts = [f"    solving… {elapsed}s"]
    if state.get("solved"):
        parts.append("SOLVED")
    elif state.get("peak_odds") is not None:
        # Only show odds once we have a meaningful peak (avoids spam at ~1.0)
        peak = float(state["peak_odds"])
        if peak > 1.5:
            parts.append(_closeness_label(peak))
    line = "  ".join(parts)
    if hint:
        line = f"{line}  — {hint}"
    return line


def _run_solve_field_streaming(
    cmd: list[str],
    *,
    env: dict[str, str],
    timeout: float,
    stream_terminal: bool = False,
    progress: bool = False,
    progress_hint: str | None = None,
) -> tuple[int, str]:
    """Run solve-field; always capture full text.

    ``stream_terminal`` echoes every line. ``progress`` updates a single status
    line with search depth / index (ignored if ``stream_terminal`` is set).
    Uses ``select`` so the wall-clock timeout is honored even when
    solve-field is silent for long stretches.
    """
    import sys

    # Line-buffer solve-field when possible so the terminal updates live
    run_cmd = list(cmd)
    if shutil.which("stdbuf"):
        run_cmd = ["stdbuf", "-oL", "-eL", *run_cmd]

    show_command(run_cmd)

    proc = subprocess.Popen(
        run_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    chunks: list[str] = []
    deadline = time.monotonic() + timeout
    timed_out = False
    fd = proc.stdout.fileno()
    t0 = time.monotonic()
    prog: dict[str, Any] = {}
    widest = 0
    is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    show_progress = bool(progress) and not stream_terminal

    last_paint = 0.0

    def _paint(*, force: bool = False) -> None:
        nonlocal widest, last_paint
        if not show_progress:
            return
        now = time.monotonic()
        # Non-TTY: at most one status line every 10s (not per index).
        if not is_tty and not force and (now - last_paint) < 10.0:
            return
        # TTY: refresh at most ~2 Hz so the line does not flicker.
        if is_tty and not force and (now - last_paint) < 0.5:
            return
        last_paint = now
        elapsed = int(now - t0)
        line = _wcs_progress_line(elapsed, prog, hint=progress_hint)
        if is_tty:
            widest = max(widest, len(line))
            print("\r" + dim(line), end="", flush=True)
        else:
            print(dim(line), flush=True)

    if show_progress:
        _paint(force=True)

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                proc.kill()
                break
            ready, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if not ready:
                if proc.poll() is not None:
                    break
                _paint()  # keep the timer moving while solve-field is silent
                continue
            line = proc.stdout.readline()
            if line == "":
                if proc.poll() is not None:
                    break
                continue
            chunks.append(line)
            if stream_terminal:
                print(dim(f"    | {line.rstrip()}"), flush=True)
            elif show_progress:
                _parse_solve_progress(line, prog)
                _paint()
        # Drain anything left after kill / exit
        rest = proc.stdout.read()
        if rest:
            chunks.append(rest)
            if stream_terminal:
                for line in rest.splitlines():
                    print(dim(f"    | {line}"), flush=True)
            elif show_progress:
                for line in rest.splitlines():
                    _parse_solve_progress(line, prog)
        returncode = proc.wait(timeout=5)
    except Exception:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        raise
    finally:
        if show_progress and is_tty:
            print("\r" + " " * max(widest, 80) + "\r", end="", flush=True)

    text = "".join(chunks).rstrip()
    if timed_out:
        # Match subprocess.TimeoutExpired style for callers
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output=text)
    return int(returncode if returncode is not None else -1), text


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
) -> tuple[str, str, str]:
    """Plate-solve one science FITS in place.

    Returns ``(status, message, solver_output)`` where status is
    ``ok`` / ``skip`` / ``fail`` and ``solver_output`` is captured
    solve-field stdout/stderr (may be empty for skip). The caller
    adds ``w`` to the filename prefix after a successful solve.
    """
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        if has_wcs(header) and not overwrite:
            return "skip", "already has WCS (pass --overwrite to re-solve)", ""
        pointing = parse_pointing(header)

    solve_bin = str(wcs_cfg.get("solve_field") or "solve-field")
    if shutil.which(solve_bin) is None and not Path(solve_bin).is_file():
        return (
            "fail",
            f"solve-field not found: {solve_bin!r} (install astrometry.net)",
            "",
        )

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
        timeout = float(wcs_cfg.get("timeout_sec") or 300)
        # System solve-field shells out to /usr/bin/python3. A newer user-site
        # NumPy (no np.bool) breaks removelines/uniformize — isolate that.
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        cmd_txt = " ".join(cmd)
        stream_terminal = bool(wcs_cfg.get("show_solver_output", False))

        try:
            if stream_terminal:
                returncode, body = _run_solve_field_streaming(
                    cmd,
                    env=env,
                    timeout=timeout,
                    stream_terminal=True,
                )
            else:
                returncode, body = _run_solve_field_streaming(
                    cmd,
                    env=env,
                    timeout=timeout,
                    progress=True,
                    progress_hint=(
                        "go get a cup of coffee, or take a walk outside "
                        "and look up at the sky"
                    ),
                )
            solver_out = f"$ {cmd_txt}\n" + body
        except subprocess.TimeoutExpired as exc:
            partial = ""
            if getattr(exc, "output", None):
                partial = str(exc.output)
            elif getattr(exc, "stdout", None):
                partial = str(exc.stdout)
            solver_out = f"$ {cmd_txt}\n[timed out after {timeout:.0f}s]\n{partial}".rstrip()
            detail = _solver_fail_summary(partial)
            msg = f"timed out after {timeout:.0f}s"
            if detail:
                msg = f"{msg} — {detail}"
            return "fail", msg, solver_out
        except FileNotFoundError:
            return "fail", f"solve-field not found: {solve_bin!r}", f"$ {cmd_txt}"

        if returncode != 0 or not out_fits.exists():
            # Also accept default .new name if --new-fits was ignored
            alt = work / f"{work_in.stem}.new"
            if alt.exists():
                out_fits = alt
            else:
                blob = solver_out
                if "np.bool" in blob or ("numpy" in blob.lower() and "bool" in blob):
                    return (
                        "fail",
                        "solve-field Python helpers crashed (NumPy incompatibility). "
                        "Pipeline sets PYTHONNOUSERSITE=1; if this persists, upgrade "
                        "astrometry.net or remove ~/.local NumPy for system python3.",
                        solver_out,
                    )
                summary = _solver_fail_summary(blob)
                if not summary:
                    tail = blob.strip().splitlines()
                    summary = tail[-1] if tail else f"exit {returncode}"
                hint = ""
                if pointing is None:
                    hint = " (no RASTRNG/DEC pointing in header)"
                return "fail", f"no solution — {summary}{hint}", solver_out

        _merge_wcs_into_science(path, out_fits)

    idx = _solver_solved_index(solver_out)
    if idx:
        with fits.open(path, mode="update", memmap=False) as hdul:
            _append_history(
                hdul[0].header,
                f"ccd_pipeline: WCS solved with index {idx}",
            )
            hdul.flush()

    bits = ["WCS written"]
    if idx:
        bits.append(f"index={idx}")
    if pointing is not None:
        bits.append(
            f"hint RA={pointing.ra.to_string(unit=u.hour, sep=':', precision=1)} "
            f"Dec={pointing.dec.to_string(unit=u.deg, sep=':', precision=0)}"
        )
    return "ok", "  ".join(bits), solver_out


def iter_science_fits(cfg: dict) -> list[Path]:
    sci_dir = Path(cfg["paths"]["output_dir"]) / "science"
    if not sci_dir.is_dir():
        return []
    return list(iter_fits_files(sci_dir))


def _object_token(name: str) -> str:
    """Filename-safe OBJECT token (same rules as science product names)."""
    from .reduce import sanitize_filename_part

    return sanitize_filename_part(name, fallback="")


def filter_science_by_objects(paths: list[Path], objects: list[str]) -> list[Path]:
    """Keep frames whose stem contains ``.{object}.`` (pipeline naming)."""
    tokens = {_object_token(o) for o in objects if str(o).strip()}
    tokens.discard("")
    if not tokens:
        return paths
    kept: list[Path] = []
    for path in paths:
        _, rest = split_science_stage_name(path.name)
        marked = f".{Path(rest).stem}."
        if any(f".{tok}." in marked for tok in tokens):
            kept.append(path)
    return kept


def _wcs_log_path(cfg: dict) -> Path:
    """``<output_dir>/wcs_solve.log`` (night reduced folder)."""
    return Path(cfg["paths"]["output_dir"]) / "wcs_solve.log"


def _wcs_frame_log_dir(cfg: dict) -> Path:
    """Per-frame solve-field transcripts: ``<output_dir>/wcs_logs/``."""
    return Path(cfg["paths"]["output_dir"]) / "wcs_logs"


def _open_wcs_log(cfg: dict) -> tuple[Path, TextIO]:
    log_path = _wcs_log_path(cfg)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path, log_path.open("a", encoding="utf-8")


def _log_line(fh: TextIO | None, line: str) -> None:
    if fh is None:
        return
    fh.write(line.rstrip() + "\n")
    fh.flush()


def _write_frame_solver_log(cfg: dict, frame_name: str, text: str) -> Path | None:
    if not text:
        return None
    log_dir = _wcs_frame_log_dir(cfg)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{Path(frame_name).stem}.solve.log"
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def solve_science_wcs(
    cfg: dict,
    *,
    limit: int | None = None,
    overwrite: bool | None = None,
    objects: list[str] | None = None,
) -> dict[str, Any]:
    """Plate-solve calibrated science frames under ``<output_dir>/science/``.

    ``objects`` (CLI) or ``wcs.include_objects`` (YAML) restrict which frames
    are attempted. Names match the OBJECT component of science filenames
    (e.g. ``APSExS-F-1`` in ``p-2017JUN29.APSExS-F-1.g.01.fits``).
    Successful solves add ``w`` to the stage prefix (``p-`` → ``pw-``,
    ``pf-`` → ``pfw-``).
    """
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
    want = list(objects) if objects else list(wcfg.get("include_objects") or [])
    if want:
        before = len(paths)
        paths = filter_science_by_objects(paths, want)
        info(f"Object filter    : {', '.join(str(o) for o in want)} ({len(paths)}/{before} frames)")
    if limit is not None:
        paths = paths[:limit]

    if paths:
        try:
            hdr0 = fits.getheader(paths[0], 0)
        except Exception:
            hdr0 = None
        if hdr0 is not None:
            cfg_scale = float(wcfg["scale_arcsec_per_pix"])
            cfg_tol = float(wcfg.get("scale_tol_frac") or 0.05)
            used = apply_header_plate_scale(wcfg, hdr0)
            header_scale = plate_scale_from_header(hdr0)
            if (
                used is not None
                and header_scale is not None
                and (
                    abs(used - cfg_scale) / max(used, 1e-6) > 0.05
                    or abs(float(wcfg.get("scale_tol_frac") or 0) - cfg_tol) > 1e-6
                )
            ):
                lo = used * (1.0 - float(wcfg["scale_tol_frac"]))
                hi = used * (1.0 + float(wcfg["scale_tol_frac"]))
                warn(
                    f"wcs.scale_arcsec_per_pix is {cfg_scale:g}″/pix but header "
                    f"PIXSCAL×binning is {header_scale:g}″/pix. Searching "
                    f"{lo:.3g}–{hi:.3g}″/pix so a stale PIXSCAL card cannot "
                    "exclude the real scale."
                )

    out_dir = Path(cfg["paths"]["output_dir"])
    sci_dir = out_dir / "science"
    wcs_flags: list[str] = []
    if limit is not None:
        wcs_flags.extend(["--limit", str(limit)])
    if wcfg.get("overwrite"):
        wcs_flags.append("--overwrite")
    for name in want:
        wcs_flags.extend(["--object", str(name)])
    show_ccd(cfg, "wcs", *wcs_flags)
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
        _log_line(log_fh, f"timeout_sec    {wcfg.get('timeout_sec')}")
        _log_line(log_fh, f"objects        {want or '(all)'}")
        _log_line(log_fh, f"limit          {limit}")
        _log_line(log_fh, f"n_frames       {len(paths)}")
        _log_line(log_fh, f"frame_logs     {_wcs_frame_log_dir(cfg)}")
        _log_line(log_fh, "-" * 72)

        if not paths:
            if want:
                warn("No science FITS matched the object filter")
                _log_line(log_fh, f"RESULT: no frames matched objects {want}")
            else:
                warn("No science FITS found — run ccd reduce first")
                _log_line(log_fh, "RESULT: no science FITS found")
            return {"ok": 0, "skip": 0, "fail": 0, "paths": [], "log": str(log_path)}

        counts = {"ok": 0, "skip": 0, "fail": 0}
        solved_paths: list[str] = []
        show_out = bool(wcfg.get("show_solver_output", False))

        for i, path in enumerate(paths, start=1):
            print()
            print(style(f"[{i}/{len(paths)}] {path.name}", S.BOLD, S.MAGENTA))
            status, msg, solver_out = solve_one_frame(
                path, wcs_cfg=wcfg, overwrite=bool(wcfg.get("overwrite"))
            )
            if status in {"ok", "skip"}:
                path = retitle_science_file(path, processed=True, wcs=True)
                remove_unprefixed_science_twin(path)
            counts[status] = counts.get(status, 0) + 1
            _log_line(log_fh, f"[{i}/{len(paths)}] {status.upper():<4}  {path.name}  {msg}")

            frame_log = None
            if solver_out:
                frame_log = _write_frame_solver_log(cfg, path.name, solver_out)
                # Keep the summary log short; full transcript is in wcs_logs/
                if frame_log is not None:
                    _log_line(log_fh, f"  frame log: {frame_log}")
                else:
                    _log_line(log_fh, f"  solve-field output ({path.name}):")
                    for line in solver_out.splitlines():
                        _log_line(log_fh, f"  | {line}")

            if status == "ok":
                print(dim(f"    {msg}"))
                solved_paths.append(str(path))
            elif status == "skip":
                warn(f"    SKIP: {msg}")
            else:
                warn(f"    FAIL: {msg}")
                if frame_log is not None:
                    info(f"    log → {frame_log}")
                elif show_out and solver_out:
                    _print_solver_output(solver_out, max_lines=30)

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
        info(f"Per-frame solve-field logs: {_wcs_frame_log_dir(cfg)}")
        return {**counts, "paths": solved_paths, "log": str(log_path)}
    finally:
        if log_fh is not None:
            log_fh.close()
