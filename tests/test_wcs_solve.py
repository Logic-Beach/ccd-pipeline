"""Unit tests for WCS helpers (no solve-field / indexes required)."""

from __future__ import annotations

from pathlib import Path

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from ccd_pipeline.wcs_solve import (
    build_solve_field_cmd,
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


if __name__ == "__main__":
    test_parse_pointing_hdi_strings()
    test_parse_pointing_numeric_degrees()
    test_parse_pointing_missing()
    test_has_wcs()
    test_build_solve_field_cmd_includes_hints_and_scale()
    test_wcs_config_defaults()
    print("all wcs tests OK")
