from ccd_pipeline.io import filter_key, resolve_filter_name


def test_filter_key():
    assert filter_key({"FILTER1": "107", "FILTER2": "204"}) == "107,204"


def test_resolve_filter_name_from_map():
    header = {"FILTER1": "107", "FILTER2": "204", "OBJECT": "NGC_6791"}
    mapping = {"107,204": "r"}
    keywords = {"filter1": "FILTER1", "filter2": "FILTER2", "object": "OBJECT"}
    assert resolve_filter_name(header, mapping, keywords) == "r"


def test_resolve_filter_name_from_dflat_object():
    header = {"FILTER1": "999", "FILTER2": "999", "OBJECT": "dflat-g"}
    keywords = {"filter1": "FILTER1", "filter2": "FILTER2", "object": "OBJECT"}
    assert resolve_filter_name(header, {}, keywords) == "g"


def test_resolve_filter_name_from_filtname():
    header = {"FILTNAME": "u", "OBJECT": "FEIGE-66"}
    keywords = {"filter1": "FILTER1", "filter2": "FILTER2", "object": "OBJECT"}
    assert resolve_filter_name(header, {}, keywords) == "u"


def test_resolve_filter_name_from_filter_card_when_wheels_corrupt():
    # 2016-era HDI science: FILTER1/2 often "Not available", FILTER still set
    header = {
        "FILTER1": "Not available",
        "FILTER2": "Not available",
        "FILTER": "z",
        "OBJECT": "Blank_Sky",
    }
    keywords = {"filter1": "FILTER1", "filter2": "FILTER2", "object": "OBJECT"}
    assert resolve_filter_name(header, {"107,204": "z"}, keywords) == "z"


def test_resolve_filter_name_prefers_valid_wheel_map_over_filter_card():
    header = {"FILTER1": "107", "FILTER2": "203", "FILTER": "i", "OBJECT": "x"}
    mapping = {"107,203": "i"}
    keywords = {"filter1": "FILTER1", "filter2": "FILTER2", "object": "OBJECT"}
    assert resolve_filter_name(header, mapping, keywords) == "i"
