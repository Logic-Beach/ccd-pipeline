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
    assert "--verbose" in cmd
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


def test_wcs_config_defaults():
    cfg = wcs_config({})
    assert cfg["scale_arcsec_per_pix"] == 0.43
    assert cfg["solver"] == "astrometry-net"
    assert cfg["enabled"] is True
    assert cfg["timeout_sec"] == 300
    assert cfg["verbose"] is True
    assert cfg["show_solver_output"] is False


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
    line = _wcs_progress_line(12, state, hint="coffee")
    assert "depth 20–30" in line
    assert "2mass-05-07" in line
    assert "24.1/30" in line or "24.07/30" in line
    assert "(12s)" in line
    assert "coffee" in line
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
    test_solver_fail_summary()
    test_filter_science_by_objects()
    test_solve_progress_depth_parsing()
    print("all wcs tests OK")
