"""Sanity figures name their axes and state the ideal shape."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ccd_pipeline.sanity import (
    diagnostic_panels,
    overscan_panels,
    save_diagnostic_figure,
    save_overscan_figure,
)
from ccd_pipeline.science_qc import save_science_sky_figure, science_sky_panels


def test_panel_text_states_the_ideal_case():
    for panel in overscan_panels() + diagnostic_panels("bias") + diagnostic_panels("flat"):
        assert panel["ideal"]
        assert panel["xlabel"]
        assert panel["ylabel"]
    assert "0 ADU" in diagnostic_panels("bias")[1]["ideal"]
    assert diagnostic_panels("flat")[1]["ideal"].endswith("1") or "1" in diagnostic_panels("flat")[1]["ideal"]
    assert "light leak" in overscan_panels()[0]["ideal"]
    assert "fringing" in science_sky_panels()[0]["ideal"]
    assert "column x" in science_sky_panels()[2]["xlabel"]


def test_figures_render_with_ideal_titles(tmp_path: Path):
    data = np.zeros((32, 40))
    bias = save_diagnostic_figure(data, tmp_path / "bias.png", title="Master bias", kind="bias")
    flat = save_diagnostic_figure(
        np.ones((32, 40)), tmp_path / "flat.png", title="Master flat", kind="flat"
    )
    band = np.concatenate([np.full(32, 1000.0), np.full(8, 10.0)])
    over = save_overscan_figure(
        {
            "flat": {
                "analysis": {
                    "overscan_col0": 32,
                    "overscan_col1": 40,
                    "column_mean": band,
                }
            }
        },
        tmp_path / "overscan.png",
    )
    sky = save_science_sky_figure(
        np.full((32, 40), 80.0),
        {
            "sky_median": 80.0,
            "sky_mad_std": 1.0,
            "row_cut": np.full(40, 80.0),
            "col_cut": np.full(32, 80.0),
        },
        tmp_path / "science.png",
        title="science",
    )
    for path in (bias, flat, over, sky):
        assert path.is_file() and path.stat().st_size > 0
