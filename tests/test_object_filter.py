"""OBJECT exclude filters (junk / rfocus / …) and HISTORY card append."""

from __future__ import annotations

import numpy as np
from astropy.nddata import CCDData
import astropy.units as u
from astropy.io.fits import Header

from ccd_pipeline.calibrate import _append_history
from ccd_pipeline.inventory import (
    filter_excluded_rows,
    is_excluded_object,
    should_skip_science_object,
)


def test_junk_always_excluded_even_without_config():
    assert is_excluded_object("junk", {})
    assert is_excluded_object("JUNK", {"science": {}})
    assert is_excluded_object("field-junk-1", {"science": {"exclude_object_substrings": []}})
    assert not is_excluded_object("NGC_6791", {})


def test_configured_excludes_merge_with_junk():
    cfg = {"science": {"exclude_object_substrings": ["rfocus", "focus"]}}
    assert is_excluded_object("rfocus", cfg)
    assert is_excluded_object("junk", cfg)  # always on
    assert not is_excluded_object("APSExS-F-1", cfg)


def test_filter_excluded_rows_drops_junk_biases():
    cfg = {"science": {"exclude_object_substrings": ["junk"]}}
    rows = [
        {"file": "a.fits", "object": "bias", "obstype": "BIAS"},
        {"file": "b.fits", "object": "junk", "obstype": "BIAS"},
        {"file": "c.fits", "object": "dflat-r", "obstype": "FLAT"},
    ]
    kept = filter_excluded_rows(rows, cfg, warn=False)
    assert [r["file"] for r in kept] == ["a.fits", "c.fits"]


def test_science_include_list():
    cfg = {
        "science": {
            "include_objects": ["NGC_6791"],
            "exclude_object_substrings": ["junk"],
        }
    }
    assert should_skip_science_object("APSExS", cfg)
    assert not should_skip_science_object("NGC_6791", cfg)
    assert should_skip_science_object("junk", cfg)


def test_append_history_no_newline_join():
    ccd = CCDData(np.ones((2, 2)), unit=u.adu, meta=Header())
    _append_history(ccd, "ccd_pipeline: overscan subtract [4097:4150,1:4150] (median)")
    _append_history(ccd, "ccd_pipeline: trim to [1:4096,1:4112]")
    hist = [str(h) for h in ccd.meta["HISTORY"]]
    assert len(hist) >= 2
    assert not any("\n" in h for h in hist)
