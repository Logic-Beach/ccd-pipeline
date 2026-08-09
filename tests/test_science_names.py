from ccd_pipeline.reduce import (
    sanitize_filename_part,
    science_output_name,
    science_product_stem,
)


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
        == "2017JUN24.SA_103-Z.r.fits"
    )


def test_repeat_object_filter_gets_seq():
    assert (
        science_output_name(
            "2017JUN24", "BD+26o2606", "r", occurrence=1, total_for_key=5
        )
        == "2017JUN24.BD+26o2606.r.01.fits"
    )
    assert (
        science_output_name(
            "2017JUN24", "BD+26o2606", "r", occurrence=5, total_for_key=5
        )
        == "2017JUN24.BD+26o2606.r.05.fits"
    )
