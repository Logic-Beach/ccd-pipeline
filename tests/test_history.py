"""HISTORY cards must be appendable without crashing FITS header validation."""

from __future__ import annotations

import numpy as np
from astropy.nddata import CCDData
import astropy.units as u
from astropy.io.fits import Header

from ccd_pipeline.calibrate import _append_history, apply_overscan_and_trim


def test_append_history_adds_discrete_cards():
    ccd = CCDData(np.ones((4, 4)), unit=u.adu, meta=Header())
    _append_history(ccd, "ccd_pipeline: first")
    _append_history(ccd, "ccd_pipeline: second")
    hist = list(ccd.meta["HISTORY"])
    assert "ccd_pipeline: first" in hist
    assert "ccd_pipeline: second" in hist
    # No newline-joined single card
    assert not any("\n" in str(h) for h in hist)


def test_overscan_and_trim_writes_two_history_cards():
    # 10x12 image: useful [1:8,1:10] → 10 rows × 8 cols; overscan cols 9–12
    data = np.full((10, 12), 1000.0)
    data[:, 8:] = 100.0  # overscan pedestal
    data[:, :8] += 50.0  # signal above pedestal
    header = Header(
        {
            "BIASSEC": "[9:12,1:10]",
            "DATASEC": "[1:8,1:10]",
        }
    )
    ccd = CCDData(data, unit=u.adu, meta=header)
    cfg = {
        "keywords": {"biassec": "BIASSEC", "datasec": "DATASEC"},
        "overscan": {"median": True},
    }
    out = apply_overscan_and_trim(ccd, cfg)
    assert out.shape == (10, 8)
    hist = [str(h) for h in out.meta["HISTORY"]]
    assert any("overscan subtract" in h for h in hist)
    assert any("trim to" in h for h in hist)
    assert not any("\n" in h for h in hist)
