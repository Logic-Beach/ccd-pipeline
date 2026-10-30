"""Dark subtraction is scaled by exposure time and skipped when disabled."""

from __future__ import annotations

import numpy as np
from astropy.nddata import CCDData
import astropy.units as u

from ccd_pipeline.calibrate import astropy_examples, darks_enabled, subtract_dark_current


def _frame(value: float, exptime: float) -> CCDData:
    return CCDData(np.full((4, 4), value), unit="adu", meta={"EXPTIME": exptime})


def test_darks_disabled_omits_subtract_dark():
    cfg = {"darks": {"enabled": False}, "keywords": {"biassec": "BIASSEC", "datasec": "DATASEC", "exptime": "EXPTIME"}, "overscan": {"median": True}}
    assert darks_enabled(cfg) is False
    assert "subtract_dark" not in " ".join(astropy_examples(cfg, kind="science"))


def test_subtract_dark_scales_to_the_frame_exposure():
    # 10 s master dark of 5 ADU is 0.5 ADU/s. A 20 s frame of 10 ADU loses 10 ADU.
    science = _frame(10.0, 20.0)
    dark = _frame(5.0, 10.0)
    out = subtract_dark_current(science, dark)
    assert np.allclose(out.data, 0.0)

    matched = subtract_dark_current(_frame(5.0, 10.0), dark)
    assert np.allclose(matched.data, 0.0)


def test_subtract_dark_without_exposure_is_unscaled():
    science = CCDData(np.full((4, 4), 9.0), unit=u.adu)
    dark = CCDData(np.full((4, 4), 4.0), unit=u.adu)
    out = subtract_dark_current(science, dark)
    assert np.allclose(out.data, 5.0)
