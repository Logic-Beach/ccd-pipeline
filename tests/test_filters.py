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
