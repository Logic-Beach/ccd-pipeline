from pathlib import Path

from ccd_pipeline.filters_discover import discover_filter_combos, _guess_name_from_object


def test_guess_name_from_object():
    assert _guess_name_from_object("dflat-r") == "r"
    assert _guess_name_from_object("dflat-Wash-M") == "Wash-M"
    assert _guess_name_from_object("NGC_6791") is None


def test_discover_on_wiyn_night():
    raw = Path("/Users/null/astronomy/data/2017JUN29/raw")
    if not raw.exists():
        return  # skip if sample data absent
    combos = discover_filter_combos(raw)
    assert len(combos) >= 10
    keys = {c["key"] for c in combos}
    assert "107,204" in keys  # r
    r = next(c for c in combos if c["key"] == "107,204")
    assert r["suggested_name"] == "r"
