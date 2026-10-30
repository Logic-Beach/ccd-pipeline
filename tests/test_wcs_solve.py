"""Unit tests for WCS helpers (no solve-field / indexes required)."""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from ccd_pipeline.wcs_solve import (
    _closeness_label,
    _parse_solve_progress,
    _short_index_name,
    _solver_fail_summary,
    _solver_solved_index,
    _wcs_progress_line,
    build_solve_field_cmd,
    filter_science_by_objects,
    has_wcs,
    parse_pointing,
    wcs_config,
)


def test_parse_pointing_hdi_strings():
    hdr = fits.Header({"RASTRNG": "+19:21:29.3", "DECSTRNG": "+37:49:14"})
    c = parse_pointing(hdr)
    assert c is not None
    assert abs(c.ra.degree - 290.372) < 0.01
    assert abs(c.dec.degree - 37.8206) < 0.01


def test_parse_pointing_numeric_degrees():
    hdr = fits.Header({"RA": 150.0, "DEC": -20.5})
    c = parse_pointing(hdr)
    assert c is not None
    assert abs(c.ra.degree - 150.0) < 1e-6
    assert abs(c.dec.degree + 20.5) < 1e-6


def test_parse_pointing_missing():
    assert parse_pointing(fits.Header({"OBJECT": "NGC_6791"})) is None


def test_has_wcs():
    assert has_wcs(fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"}))
    assert not has_wcs(fits.Header({"OBJECT": "x"}))
    assert not has_wcs(fits.Header({"CTYPE1": "PIXEL", "CRVAL1": 1}))


def test_build_solve_field_cmd_includes_hints_and_scale():
    work = Path("/tmp")
    image = work / "frame.fits"
    out = work / "frame.wcs.fits"
    pointing = SkyCoord(ra=290.37 * u.deg, dec=37.82 * u.deg, frame="icrs")
    cfg = wcs_config(
        {
            "wcs": {
                "solve_field": "/opt/solve-field",
                "scale_arcsec_per_pix": 0.43,
                "scale_tol_frac": 0.05,
                "radius_deg": 1.0,
                "downsample": 2,
                "index_dir": "/data/indexes",
            }
        }
    )
    cmd = build_solve_field_cmd(
        image, out_fits=out, work_dir=work, wcs_cfg=cfg, pointing=pointing
    )
    assert cmd[0] == "/opt/solve-field"
    assert "--no-remove-lines" in cmd
    assert "--uniformize" in cmd and cmd[cmd.index("--uniformize") + 1] == "0"
    assert "--verbose" not in cmd
    cfg_verbose = dict(cfg)
    cfg_verbose["verbose"] = True
    cmd_v = build_solve_field_cmd(
        image, out_fits=out, work_dir=work, wcs_cfg=cfg_verbose, pointing=pointing
    )
    assert "--verbose" in cmd_v
    assert "--cpulimit" in cmd
    assert "--ra" in cmd and "--dec" in cmd and "--radius" in cmd
    assert "--scale-units" in cmd and cmd[cmd.index("--scale-units") + 1] == "arcsecperpix"

    assert "--scale-low" in cmd and "--scale-high" in cmd
    lo = float(cmd[cmd.index("--scale-low") + 1])
    hi = float(cmd[cmd.index("--scale-high") + 1])
    assert abs(lo - 0.43 * 0.95) < 1e-6
    assert abs(hi - 0.43 * 1.05) < 1e-6
    assert "--downsample" in cmd and cmd[cmd.index("--downsample") + 1] == "2"
    assert "--index-dir" in cmd and cmd[cmd.index("--index-dir") + 1] == "/data/indexes"
    assert cmd[-1] == str(image)


def test_s2kb_plate_scale_includes_binning():
    from ccd_pipeline.wcs_solve import plate_scale_arcsec

    assert plate_scale_arcsec(detector="s2kb") == 0.60
    assert plate_scale_arcsec(detector="s2kb", ccdsum="2 2") == 1.20
    assert plate_scale_arcsec(instrument="hdi", ccdsum="1 1") == 0.43


def test_hall_plate_scale_not_hdi_fallback():
    from ccd_pipeline.wcs_solve import (
        apply_header_plate_scale,
        plate_scale_arcsec,
        plate_scale_from_header,
    )

    assert abs(plate_scale_arcsec(telescope="hall") - 0.369) < 1e-9
    assert abs(plate_scale_arcsec(telescope="hall", ccdsum="2 2") - 0.738) < 1e-9
    assert abs(plate_scale_arcsec(detector="E2V CCD-231 4096x4112", ccdsum="2 2") - 0.738) < 1e-9
    hdr = fits.Header({"PIXSCAL": 0.327, "CCDSUM": "2 2"})
    assert abs(plate_scale_from_header(hdr) - 0.654) < 1e-9
    # Stale PIXSCAL must not override the named Hall scale.
    assert abs(plate_scale_arcsec(telescope="hall", ccdsum="2 2", header=hdr) - 0.738) < 1e-9
    wcfg = {"scale_arcsec_per_pix": 0.86, "scale_tol_frac": 0.05}
    apply_header_plate_scale(wcfg, hdr)
    lo = wcfg["scale_arcsec_per_pix"] * (1.0 - wcfg["scale_tol_frac"])
    hi = wcfg["scale_arcsec_per_pix"] * (1.0 + wcfg["scale_tol_frac"])
    assert lo <= 0.739 <= hi
    assert lo <= 0.654 <= hi
    assert lo <= 0.86 <= hi


def test_show_command_uses_its_own_color(capsys, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    from ccd_pipeline.term import S, show_command

    show_command(["solve-field", "--ra", "10", "my frame.fits"])
    out = capsys.readouterr().out
    assert out.startswith(S.BOLD + S.BRIGHT_BLUE) or S.BRIGHT_BLUE in out
    assert "$" in out
    assert "solve-field" in out
    assert "my frame.fits" in out


def test_solve_field_runner_prints_command_first(capsys, monkeypatch):
    import os
    import sys

    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    from ccd_pipeline.wcs_solve import _run_solve_field_streaming

    code, text = _run_solve_field_streaming(
        [sys.executable, "-c", "print('solved-ok')"],
        env=os.environ.copy(),
        timeout=30,
        progress=False,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "solved-ok" in text
    assert "$" in out
    assert sys.executable in out
    # The command line is printed before the process output is returned.
    assert out.index("$") < len(out)


def test_processing_plan_shows_ccd_command(capsys, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    from ccd_pipeline.calibrate import print_processing_plan
    from ccd_pipeline.term import S

    print_processing_plan(
        {
            "_config_path": "/tmp/night.yaml",
            "fits": {"image_hdu": 0, "unit": "adu"},
            "keywords": {"biassec": "BIASSEC", "datasec": "DATASEC"},
            "overscan": {"median": True},
            "darks": {"enabled": False},
        },
        kind="bias",
    )
    out = capsys.readouterr().out
    assert S.BRIGHT_BLUE in out
    assert "ccd masters --config /tmp/night.yaml --only bias" in out
    assert "$ 1." not in out
    assert "overscan" in out
    underlying = next(line for line in out.splitlines() if "ccdproc.subtract_overscan" in line)
    assert S.DIM in underlying and S.BLUE in underlying
    assert S.BRIGHT_BLUE not in underlying
    assert "fits_section=header['BIASSEC']" in underlying
    assert "ccdproc.combine" in out


def test_flat_plan_shows_ccdproc_under_pipeline_command(capsys, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    from ccd_pipeline.calibrate import print_processing_plan

    print_processing_plan(
        {
            "_config_path": "/Volumes/luna/2013_OCT/configs/test_2013oct12.yaml",
            "fits": {"image_hdu": 0, "unit": "adu"},
            "keywords": {"biassec": "BIASSEC", "datasec": "DATASEC"},
            "overscan": {"median": True, "biassec": "[1036:1056,1:1024]"},
            "darks": {"enabled": False},
            "combine": {
                "flat": {
                    "method": "average",
                    "scale": "inv_median",
                    "sigma_clip": True,
                    "sigma_clip_low": 5,
                    "sigma_clip_high": 5,
                }
            },
        },
        kind="flat",
    )
    out = capsys.readouterr().out
    ccd_at = out.index("ccd masters")
    overscan_at = out.index("ccdproc.subtract_overscan")
    assert ccd_at < overscan_at
    assert "fits_section='[1036:1056,1:1024]'" in out
    assert "ccdproc.subtract_bias(ccd, master_bias)" in out
    assert "scale=inv_median" in out
    assert "sigma_clip_low_thresh=5" in out


def test_prompt_prints_rule_after_answer(capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    from ccd_pipeline.prompt import ask_yes_no

    assert ask_yes_no("Build master bias", True) is True
    assert "-------------------------------------------------------" in capsys.readouterr().out


def test_wcs_config_defaults():
    cfg = wcs_config({})
    assert cfg["scale_arcsec_per_pix"] == 0.43
    assert cfg["solver"] == "astrometry-net"
    assert cfg["enabled"] is True
    assert cfg["timeout_sec"] == 300
    assert cfg["verbose"] is False
    assert cfg["show_solver_output"] is False


def test_solver_solved_index():
    text = (
        "Trying index /usr/share/astrometry/index-2mass-05-07.fits...\n"
        "Field 1: solved with index index-2mass-05-07.fits.\n"
    )
    assert _solver_solved_index(text) == "2mass-05-07"
    assert _solver_solved_index("") is None


def test_solver_fail_summary():
    text = (
        "Field 1 did not solve (index index-2mass-07-03.fits, field objects 191-200).\n"
        "Best odds encountered: 1.54397\n"
        "Trying index /usr/share/astrometry/index-2mass-05-07.fits...\n"
        "Got a new best match: logodds 3.18109.\n"
        "Best odds encountered: 24.0729\n"
        "Trying index /usr/share/astrometry/index-2mass-05-07.fits...\n"
        "Best odds encountered: 1\n"
        "Field 1 did not solve (index index-2mass-05-07.fits, field objects 1-10).\n"
    )
    s = _solver_fail_summary(text)
    assert "24" in s
    assert "need ~30" in s
    assert "index-2mass-05-07" in s


def test_filter_science_by_objects():
    paths = [
        Path("2017JUN29.APSExS-F-1.g.01.fits"),
        Path("2017JUN29.APSExS-F-3.g.01.fits"),
        Path("2017JUN29.NGC_6791.r.fits"),
    ]
    got = filter_science_by_objects(paths, ["APSExS-F-1", "NGC_6791"])
    assert [p.name for p in got] == [
        "2017JUN29.APSExS-F-1.g.01.fits",
        "2017JUN29.NGC_6791.r.fits",
    ]
    assert filter_science_by_objects(paths, []) == paths


def test_solve_progress_depth_parsing():
    state: dict = {}
    assert _parse_solve_progress("startdepth 20", state)
    assert _parse_solve_progress("enddepth 30", state)
    assert state["depth_lo"] == 20 and state["depth_hi"] == 30
    assert _parse_solve_progress(
        "Trying index /usr/share/astrometry/index-2mass-05-07.fits...", state
    )
    assert state["index"] == "2mass-05-07"
    assert _short_index_name("index-tycho2-09.littleendian.fits") == "tycho2-09"
    assert _parse_solve_progress("Best odds encountered: 24.0729", state)
    assert state["peak_odds"] == 24.0729
    line = _wcs_progress_line(12, state)
    assert "solving…" in line
    assert "12s" in line
    assert "24.1/30" in line or "24.07/30" in line
    assert "depth" not in line  # keep the live line short
    assert _parse_solve_progress(
        "Field 1: solved with index index-2mass-05-07.fits.", state
    )
    assert state["solved"] is True
    assert state["solved_index"] == "2mass-05-07"
    assert _closeness_label(None).startswith("match —")
    assert "far" in _closeness_label(1.0)
    assert "strong" in _closeness_label(114.0)
    assert "strong" in _closeness_label(35.0)


if __name__ == "__main__":
    test_parse_pointing_hdi_strings()
    test_parse_pointing_numeric_degrees()
    test_parse_pointing_missing()
    test_has_wcs()
    test_build_solve_field_cmd_includes_hints_and_scale()
    test_wcs_config_defaults()
    test_solver_solved_index()
    test_solver_fail_summary()
    test_filter_science_by_objects()
    test_solve_progress_depth_parsing()
    print("all wcs tests OK")
