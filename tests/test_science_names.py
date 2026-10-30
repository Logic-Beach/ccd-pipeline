from pathlib import Path

from ccd_pipeline.reduce import (
    remove_unprefixed_science_twin,
    retitle_science_file,
    sanitize_filename_part,
    science_output_name,
    science_product_stem,
    science_stage_tag,
    split_science_stage_name,
    with_science_stages,
)
from ccd_pipeline.science_qc import parse_science_filename
from ccd_pipeline.wcs_solve import filter_science_by_objects


def test_sanitize_spaces_and_junk():
    assert sanitize_filename_part("rfocus T=23.8 C") == "rfocus_T_23.8_C"
    assert sanitize_filename_part("BD+26o2606") == "BD+26o2606"
    assert sanitize_filename_part("SA_103-Z") == "SA_103-Z"
    assert sanitize_filename_part("  ") == "unknown"


def test_stem_basic():
    assert science_product_stem("2017JUN24", "SA_103-Z", "r") == "2017JUN24.SA_103-Z.r"


def test_unique_object_filter_no_seq():
    assert (
        science_output_name(
            "2017JUN24", "SA_103-Z", "r", occurrence=1, total_for_key=1
        )
        == "p-2017JUN24.SA_103-Z.r.fits"
    )


def test_repeat_object_filter_gets_seq():
    assert (
        science_output_name(
            "2017JUN24", "BD+26o2606", "r", occurrence=1, total_for_key=5
        )
        == "p-2017JUN24.BD+26o2606.r.01.fits"
    )
    assert (
        science_output_name(
            "2017JUN24", "BD+26o2606", "r", occurrence=5, total_for_key=5
        )
        == "p-2017JUN24.BD+26o2606.r.05.fits"
    )


def test_stage_tag_order():
    assert science_stage_tag(processed=True) == "p"
    assert science_stage_tag(processed=True, fringe=True) == "pf"
    assert science_stage_tag(processed=True, wcs=True) == "pw"
    assert science_stage_tag(processed=True, fringe=True, wcs=True) == "pfw"


def test_split_and_with_science_stages():
    bare = Path("2017JUN24.SA_103-Z.r.fits")
    assert split_science_stage_name(bare.name) == ("", "2017JUN24.SA_103-Z.r.fits")
    processed = with_science_stages(bare, processed=True)
    assert processed.name == "p-2017JUN24.SA_103-Z.r.fits"
    assert with_science_stages(processed, fringe=True).name == "pf-2017JUN24.SA_103-Z.r.fits"
    assert with_science_stages(processed, wcs=True).name == "pw-2017JUN24.SA_103-Z.r.fits"
    assert (
        with_science_stages(processed, fringe=True, wcs=True).name
        == "pfw-2017JUN24.SA_103-Z.r.fits"
    )
    assert (
        with_science_stages(Path("pw-2017JUN24.SA_103-Z.r.fits"), fringe=True).name
        == "pfw-2017JUN24.SA_103-Z.r.fits"
    )
    assert split_science_stage_name("pfw-2017JUN24.SA_103-Z.r.fits") == (
        "pfw",
        "2017JUN24.SA_103-Z.r.fits",
    )


def test_parse_science_filename_strips_stage_prefix():
    meta = parse_science_filename(Path("pfw-2017JUN24.SA_103-Z.r.01.fits"))
    assert meta["night"] == "2017JUN24"
    assert meta["object"] == "SA_103-Z"
    assert meta["filter"] == "r"
    assert meta["seq"] == "01"
    assert meta["stages"] == "pfw"
    bare = parse_science_filename(Path("2017JUN24.SA_103-Z.r.fits"))
    assert bare["object"] == "SA_103-Z"
    assert bare["stages"] == ""


def test_filter_science_by_objects_ignores_stage_prefix():
    paths = [
        Path("p-2017JUN29.APSExS-F-1.g.01.fits"),
        Path("pf-2017JUN29.APSExS-F-3.g.01.fits"),
        Path("pw-2017JUN29.NGC_6791.r.fits"),
    ]
    got = filter_science_by_objects(paths, ["APSExS-F-1", "NGC_6791"])
    assert [p.name for p in got] == [
        "p-2017JUN29.APSExS-F-1.g.01.fits",
        "pw-2017JUN29.NGC_6791.r.fits",
    ]


def test_retitle_science_file_renames(tmp_path):
    src = tmp_path / "p-2017JUN24.SA_103-Z.r.fits"
    src.write_bytes(b"not-fits")
    dest = retitle_science_file(src, fringe=True)
    assert dest.name == "pf-2017JUN24.SA_103-Z.r.fits"
    assert dest.exists()
    assert not src.exists()
    same = retitle_science_file(dest, fringe=True)
    assert same == dest


def test_remove_unprefixed_science_twin(tmp_path):
    legacy = tmp_path / "2017JUN24.SA_103-Z.r.fits"
    prefixed = tmp_path / "p-2017JUN24.SA_103-Z.r.fits"
    legacy.write_bytes(b"old")
    prefixed.write_bytes(b"new")
    removed = remove_unprefixed_science_twin(prefixed)
    assert removed == legacy
    assert not legacy.exists()
    assert prefixed.exists()
    assert remove_unprefixed_science_twin(prefixed) is None
