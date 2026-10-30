"""Overscan region override: skip light-leaked columns next to the image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from astropy.io.fits import Header
from astropy.nddata import CCDData
import astropy.units as u

from ccd_pipeline.calibrate import apply_overscan_and_trim
from ccd_pipeline.night_init import save_overscan_biassec
from ccd_pipeline.sanity import (
    analyze_overscan,
    check_overscan,
    clip_section_y,
    crop_section_rows,
    row_settle_counts,
    section_inside,
    shrink_biassec,
    suggest_edge_drop,
    suggest_row_crop,
)


def test_shrink_right_overscan_drops_image_edge():
    header = "[1025:1056,1:1024]"
    data = "[1:1024,1:1024]"
    assert shrink_biassec(header, data, 5) == "[1030:1056,1:1024]"


def test_shrink_left_overscan_drops_image_edge():
    assert shrink_biassec("[1:32,1:100]", "[33:200,1:100]", 4) == "[1:28,1:100]"


def test_section_inside_header_overscan():
    outer = "[1025:1056,1:1024]"
    assert section_inside("[1030:1056,1:1024]", outer)
    assert not section_inside("[1024:1056,1:1024]", outer)


def test_trimmed_overscan_drop_uses_image_edge():
    data = np.full((8, 16), 1000.0)
    data[:, 8:12] = 400.0  # gap left behind when the overscan is trimmed
    data[:, 12:16] = 50.0  # usable overscan
    analysis = analyze_overscan(data, "[13:16,1:8]", "[1:8,1:8]")
    assert analysis["useful_edge_median"] > 500
    notes = check_overscan(analysis, kind="flat")
    assert any(level == "ok" and "sharp drop" in msg for level, msg in notes)
    assert not any("weak drop" in msg for _, msg in notes)


def test_suggest_edge_drop_stops_at_far_side_level():
    values = [80, 40, 12, 6, 5, 5, 5, 5, 5, 5, 5, 5]
    # 80, 40, and 12 sit more than 5 ADU above the far-side level (~5).
    assert suggest_edge_drop(values) == 3


def test_override_biassec_ignores_leaked_columns():
    data = np.full((10, 12), 1000.0)
    data[:, 8:10] = 500.0  # FITS columns 9–10, next to the image
    data[:, 10:12] = 100.0  # FITS columns 11–12, clean
    ccd = CCDData(
        data,
        unit=u.adu,
        meta=Header({"BIASSEC": "[9:12,1:10]", "DATASEC": "[1:8,1:10]"}),
    )
    cfg = {
        "keywords": {"biassec": "BIASSEC", "datasec": "DATASEC"},
        "overscan": {"median": True, "biassec": "[11:12,1:10]"},
    }
    out = apply_overscan_and_trim(ccd, cfg)
    assert out.shape == (10, 8)
    assert abs(float(np.median(out.data)) - 900.0) < 1
    hist = [str(card) for card in out.meta["HISTORY"]]
    assert any("[11:12,1:10]" in card for card in hist)


def test_row_settle_counts_ignores_a_smooth_slope():
    y = 1600.0 + 0.05 * np.arange(200)
    n_lo, n_hi, _typical = row_settle_counts(y)
    assert n_lo == 0 and n_hi == 0


def test_row_settle_counts_finds_leading_hook():
    y = np.full(100, 1622.0)
    y[:8] = [472, 900, 1200, 1400, 1500, 1560, 1590, 1610]
    n_lo, n_hi, typical = row_settle_counts(y)
    assert n_lo == 8
    assert n_hi == 0
    assert abs(typical - 1622) < 1


def test_crop_section_rows_drops_readout_start():
    assert crop_section_rows("[2080:2124,1:2056]", n_lo=15) == "[2080:2124,16:2056]"
    assert suggest_row_crop(12, 2056) == 15


def test_check_overscan_warns_on_y_settling_not_a_slope():
    data = np.full((40, 20), 1622.0)
    data[:8, :] = 500.0
    analysis = analyze_overscan(data, "[13:20,1:40]", "[1:12,1:40]")
    assert analysis["y_settle_lo"] >= 8
    notes = check_overscan(analysis, kind="bias")
    text = " ".join(msg for _, msg in notes)
    assert "settling" in text
    assert "varies with row" not in text
    cropped = analyze_overscan(data, "[13:20,1:40]", "[1:12,9:40]")
    assert cropped["y_settle_lo"] == 0


def test_short_biassec_y_does_not_crash_and_trims():
    """User-entered [x1:x2,y0:y1] must not be passed raw to ccdproc."""
    data = np.full((20, 12), 1622.0)
    data[:5, :] = 400.0
    ccd = CCDData(
        data,
        unit=u.adu,
        meta=Header({"BIASSEC": "[9:12,1:20]", "DATASEC": "[1:8,1:20]"}),
    )
    cfg = {
        "keywords": {"biassec": "BIASSEC", "datasec": "DATASEC"},
        "overscan": {
            "median": True,
            "model": "none",
            "biassec": "[9:12,6:20]",
        },
    }
    out = apply_overscan_and_trim(ccd, cfg)
    assert out.shape == (15, 8)
    assert abs(float(np.median(out.data))) < 1


def test_legendre_fit_uses_only_kept_overscan_rows():
    """Rows outside BIASSEC must not be samples in the polynomial."""
    ny, nx = 20, 12
    rows = np.arange(ny, dtype=float)
    data = np.repeat((1000.0 + 10.0 * rows)[:, None], nx, axis=1)
    data[:5, :] = 100.0
    data[:, :8] += 25.0
    original = data.copy()
    ccd = CCDData(
        data,
        unit=u.adu,
        meta=Header({"BIASSEC": "[9:12,1:20]", "DATASEC": "[1:8,1:20]"}),
    )
    cfg = {
        "keywords": {"biassec": "BIASSEC", "datasec": "DATASEC"},
        "overscan": {
            "median": True,
            "model": "legendre",
            "order": 1,
            "biassec": "[9:12,6:20]",
        },
    }
    out = apply_overscan_and_trim(ccd, cfg)
    assert out.shape == (15, 8)
    assert np.allclose(out.data, 25.0)
    assert np.array_equal(ccd.data, original)


def test_clip_section_y_keeps_trim_x():
    assert clip_section_y("[26:2074,1:2056]", "[2085:2120,20:2056]") == (
        "[26:2074,20:2056]"
    )


def test_datasec_override_trims_settling_rows():
    data = np.full((20, 12), 1622.0)
    data[:5, :] = 400.0
    ccd = CCDData(
        data,
        unit=u.adu,
        meta=Header({"BIASSEC": "[9:12,1:20]", "DATASEC": "[1:8,1:20]"}),
    )
    cfg = {
        "keywords": {"biassec": "BIASSEC", "datasec": "DATASEC"},
        "overscan": {
            "median": True,
            "model": "none",
            "biassec": "[9:12,1:20]",
            "datasec": "[1:8,6:20]",
        },
    }
    out = apply_overscan_and_trim(ccd, cfg)
    assert out.shape == (15, 8)
    # Settling rows are gone; remaining image − overscan ≈ 0.
    assert abs(float(np.median(out.data))) < 1


def test_save_overscan_biassec_roundtrip(tmp_path: Path):
    path = tmp_path / "night.yaml"
    path.write_text(
        "# generated\n"
        "overscan:\n"
        "  median: true\n"
        "night_id: x\n"
    )
    cfg: dict = {"overscan": {"median": True}}
    save_overscan_biassec(path, cfg, "[1030:1056,1:1024]")
    saved = yaml.safe_load(path.read_text())
    assert saved["overscan"]["biassec"] == "[1030:1056,1:1024]"
    assert saved["overscan"]["median"] is True
    assert cfg["overscan"]["biassec"] == "[1030:1056,1:1024]"
    assert path.read_text().startswith("# generated\n")

    save_overscan_biassec(
        path, cfg, "[2080:2124,16:2056]", datasec="[26:2074,16:2056]", update_datasec=True
    )
    saved = yaml.safe_load(path.read_text())
    assert saved["overscan"]["datasec"] == "[26:2074,16:2056]"
    assert cfg["overscan"]["datasec"] == "[26:2074,16:2056]"

    save_overscan_biassec(path, cfg, None)
    saved = yaml.safe_load(path.read_text())
    assert "biassec" not in saved["overscan"]
    assert "biassec" not in cfg["overscan"]
