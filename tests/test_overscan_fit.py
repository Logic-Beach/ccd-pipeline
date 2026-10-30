"""Overscan level comes from a fitted low-order model, not each row's own median."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.nddata import CCDData

from ccd_pipeline.calibrate import (
    apply_overscan_and_trim,
    astropy_examples,
    describe_frame_steps,
    describe_overscan_estimate,
    exposure_seconds,
    overscan_model,
    overscan_settings,
)


def _cfg(**overscan):
    return {
        "keywords": {"biassec": "BIASSEC", "datasec": "DATASEC", "exptime": "EXPTIME"},
        "fits": {"image_hdu": 0, "unit": "adu"},
        "overscan": {"median": True, **overscan},
        "darks": {"enabled": False},
        "combine": {},
    }


def _noisy_frame(seed: int = 0) -> CCDData:
    """1024 rows, 1024 image columns + 32 overscan columns, 6 ADU read noise,
    and a 10 ADU bias ramp from bottom to top that the overscan shares."""
    rng = np.random.default_rng(seed)
    ny, nx = 1024, 1056
    ramp = np.linspace(0.0, 10.0, ny)[:, None]
    data = 1600.0 + ramp + rng.normal(0.0, 6.0, (ny, nx))
    return CCDData(
        data, unit="adu", meta={"BIASSEC": "[1025:1056,1:1024]", "DATASEC": "[1:1024,1:1024]"}
    )


def _row_scatter(ccd: CCDData) -> float:
    return float(np.std(np.median(ccd.data[:, 100:1000], axis=1)))


def test_default_is_a_legendre_line():
    assert overscan_settings(_cfg()) == {"median": True, "model": "legendre", "order": 1}
    model = overscan_model(_cfg())
    assert type(model).__name__ == "Legendre1D" and model.degree == 1
    assert overscan_model(_cfg(model="none")) is None
    assert overscan_settings(_cfg(model="chebyshev", order=0))["model"] == "chebyshev"
    with pytest.raises(ValueError):
        overscan_settings(_cfg(model="spline"))


def test_fit_does_not_print_overscan_noise_onto_rows():
    frame = _noisy_frame()
    fitted = apply_overscan_and_trim(frame, _cfg())
    per_row = apply_overscan_and_trim(frame, _cfg(model="none"))
    floor = 1.25 * 6.0 / np.sqrt(900)  # a 900-pixel row median of 6 ADU noise
    assert _row_scatter(fitted) < 1.5 * floor
    assert _row_scatter(per_row) > 4 * floor
    # Both remove the shared ramp and leave the image near zero
    assert abs(np.median(fitted.data)) < 0.5
    assert fitted.shape == (1024, 1024)


def test_plan_and_history_name_the_estimate():
    cfg = _cfg(biassec="[1036:1056,1:1024]")
    call = astropy_examples(cfg, kind="bias")[0]
    assert "model=models.Legendre1D(1)" in call and "median=True" in call
    assert "model=" not in astropy_examples(_cfg(model="none"), kind="bias")[0]
    step = describe_frame_steps(cfg, kind="bias")[1]
    assert "legendre straight line" in step
    assert describe_overscan_estimate(_cfg(model="legendre", order=0)).endswith("constant fitted along the rows")
    out = apply_overscan_and_trim(_noisy_frame(), cfg)
    assert any("legendre" in str(h) for h in out.meta["HISTORY"])


def test_exposure_seconds_falls_back_to_darktime():
    assert exposure_seconds({"EXPTIME": 0.0, "DARKTIME": 600.0}) == 600.0
    assert exposure_seconds({"EXPTIME": 8.0, "DARKTIME": 8.0}) == 8.0
    assert exposure_seconds({"EXPTIME": 0.0}) is None
