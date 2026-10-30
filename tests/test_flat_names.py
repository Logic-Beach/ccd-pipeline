"""Master-flat filenames: jc_ / sdss_ tags for UBVRI vs ugriz."""

from __future__ import annotations

from ccd_pipeline.io import (
    filter_filename_token,
    filter_name_from_master_flat_path,
    filter_name_from_token,
    master_flat_path,
)


def test_jc_vs_sdss_tokens():
    assert filter_filename_token("I") == "jc_I"
    assert filter_filename_token("i") == "sdss_i"
    assert filter_filename_token("R") == "jc_R"
    assert filter_filename_token("r") == "sdss_r"
    assert filter_filename_token("V") == "jc_V"
    assert filter_filename_token("u") == "sdss_u"
    assert filter_filename_token("I") != filter_filename_token("i")


def test_other_filters_unchanged():
    assert filter_filename_token("Wash-C") == "Wash-C"
    assert filter_filename_token("OIII") == "OIII"
    assert filter_filename_token("6580") == "6580"
    assert filter_filename_token("Wash-51") == "Wash-51"


def test_token_roundtrip():
    for name in ("I", "i", "R", "r", "V", "g", "Wash-51", "6580", "OIII"):
        assert filter_name_from_token(filter_filename_token(name)) == name


def test_master_flat_paths_distinct(tmp_path):
    p_I = master_flat_path(tmp_path, "I")
    p_i = master_flat_path(tmp_path, "i")
    assert p_I.name == "master_flat_jc_I.fits"
    assert p_i.name == "master_flat_sdss_i.fits"
    assert p_I.name.lower() != p_i.name.lower()


def test_name_from_path_legacy_and_header(tmp_path):
    legacy = tmp_path / "master_flat_g.fits"
    assert filter_name_from_master_flat_path(legacy) == "g"
    coded = tmp_path / "master_flat_jc_I.fits"
    assert filter_name_from_master_flat_path(coded) == "I"
    assert filter_name_from_master_flat_path(coded, {"FILTER": "I"}) == "I"
    # Older uc_/lc_ tokens still decode
    assert filter_name_from_token("uc_I") == "I"
    assert filter_name_from_token("lc_i") == "i"
