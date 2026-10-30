from pathlib import Path

from astropy.io import fits

from ccd_pipeline.filters_discover import discover_filter_combos, _guess_name_from_object
from ccd_pipeline.inventory import is_flat_obstype
from ccd_pipeline.io import iter_fits_files
from ccd_pipeline.night_init import inspect_raw_dir


def test_guess_name_from_object():
    assert _guess_name_from_object("dflat-r") == "r"
    assert _guess_name_from_object("dflat-Wash-M") == "Wash-M"
    assert _guess_name_from_object("oct12-dflat-u") == "u"
    assert _guess_name_from_object("oct12-tflat-i") == "i"
    assert _guess_name_from_object("NGC_6791") is None


def _flat(path: Path, *, obj: str, f1: str, f2: str) -> None:
    fits.PrimaryHDU(
        header=fits.Header(
            {
                "OBSTYPE": "FLAT",
                "OBJECT": obj,
                "FILTER1": f1,
                "FILTER2": f2,
                "EXPTIME": 1.0,
            }
        )
    ).writeto(path, overwrite=True)


def test_discover_reads_band_inside_object_name(tmp_path: Path):
    _flat(tmp_path / "a.fits", obj="oct12-dflat-u", f1="7", f2="2")
    _flat(tmp_path / "b.fits", obj="oct12-tflat-u", f1="7", f2="2")
    combos = discover_filter_combos(tmp_path)
    assert len(combos) == 1
    assert combos[0]["suggested_name"] == "u"
    assert combos[0]["named_from_object"] is True


def test_discover_flags_missing_band(tmp_path: Path):
    _flat(tmp_path / "a.fits", obj="oct12-pm-flat", f1="7", f2="3")
    combos = discover_filter_combos(tmp_path)
    assert combos[0]["suggested_name"] == "7_3"
    assert combos[0]["named_from_object"] is False


def test_discover_flags_disagreeing_bands(tmp_path: Path):
    _flat(tmp_path / "a.fits", obj="oct12-dflat-u", f1="7", f2="2")
    _flat(tmp_path / "b.fits", obj="oct12-dflat-g", f1="7", f2="2")
    _flat(tmp_path / "c.fits", obj="oct12-dflat-g", f1="7", f2="2")
    combos = discover_filter_combos(tmp_path)
    assert combos[0]["suggested_name"] == "g"
    assert combos[0]["named_from_object"] is False


def test_discover_skips_dotfiles(tmp_path: Path):
    real = tmp_path / "flat.fits"
    fits.PrimaryHDU(
        header=fits.Header(
            {
                "OBSTYPE": "FLAT",
                "OBJECT": "dflat-r",
                "FILTER1": "7",
                "FILTER2": "4",
                "EXPTIME": 1.0,
            }
        )
    ).writeto(real, overwrite=True)
    # AppleDouble sidecar and other dotfiles match *.fits but are not FITS.
    (tmp_path / "._flat.fits").write_bytes(b"not a fits")
    (tmp_path / ".hidden.fits").write_bytes(b"not a fits")
    (tmp_path / "latest.fits").write_bytes(b"placeholder")

    assert [p.name for p in iter_fits_files(tmp_path)] == ["flat.fits"]
    combos = discover_filter_combos(tmp_path)
    assert len(combos) == 1
    assert combos[0]["example_file"] == "flat.fits"
    assert combos[0]["n"] == 1


def test_skyflat_and_filtname_are_discovered(tmp_path: Path):
    fits.PrimaryHDU(
        header=fits.Header(
            {
                "OBSTYPE": "SKYFLAT",
                "OBJECT": "u_flat",
                "FILTNAME": "u",
                "EXPTIME": 0.5,
            }
        )
    ).writeto(tmp_path / "flat.fits", overwrite=True)
    combos = discover_filter_combos(tmp_path, flat_value="FLAT")
    assert len(combos) == 1
    assert combos[0]["suggested_name"] == "u"
    assert combos[0]["named_from_object"] is True


def test_is_flat_obstype_accepts_skyflat():
    assert is_flat_obstype("SKYFLAT", "FLAT")
    assert is_flat_obstype("DOMEFLAT")
    assert is_flat_obstype("FLAT", "FLAT")
    assert not is_flat_obstype("OBJECT", "FLAT")
    assert not is_flat_obstype("BIAS")


def test_inspect_raw_dir_hall_headers(tmp_path: Path):
    fits.PrimaryHDU(
        data=[[1, 2], [3, 4]],
        header=fits.Header(
            {
                "TELESCOP": "hall",
                "DETECTOR": "E2V CCD-231",
                "OBSTYPE": "SKYFLAT",
                "OBJECT": "g_flat",
                "FILTNAME": "g",
                "EXPTIME": 1.0,
                "BIASSEC": "[10:12,1:2]",
                "TRIMSEC": "[1:8,1:2]",
            }
        ),
    ).writeto(tmp_path / "flat.fits", overwrite=True)
    info = inspect_raw_dir(tmp_path)
    assert info["obstypes"]["flat"] == "SKYFLAT"
    assert info["keywords"]["filter1"] == "FILTNAME"
    assert info["filter_map"] == {"g,g": "g"}


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
