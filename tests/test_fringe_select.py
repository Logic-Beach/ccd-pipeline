"""Fringe frame selection modes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from ccd_pipeline.fringe import (
    _frame_mjd,
    collect_fringe_catalog,
    filter_similar_conditions,
    format_fringe_catalog,
    fringe_config,
    offer_fringe_frames,
    select_fringe_inputs,
)
from ccd_pipeline.prompt import ask_index_picks, choose


def _touch_sci(path: Path, *, obj: str, airmass: float, mjd: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Filename convention: night.object.filter.fits
    fits.PrimaryHDU(
        data=None,
        header=fits.Header(
            {
                "OBJECT": obj,
                "AIRMASS": airmass,
                "MJD-OBS": mjd,
                "FILTER": "z",
            }
        ),
    ).writeto(path, overwrite=True)
    # Empty primary is awkward for getdata; header-only is fine for selection tests
    return path


def _write_sci(path: Path, *, obj: str, **cards) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sky = float(cards.pop("sky", 100.0))
    hdr = {
        "OBJECT": obj,
        "FILTER": "z",
        "EXPTIME": 30.0,
        **cards,
    }
    fits.PrimaryHDU(data=np.full((8, 8), sky), header=fits.Header(hdr)).writeto(
        path, overwrite=True
    )
    return path


def test_fringe_config_select_modes():
    assert fringe_config({"fringe": {"select": "all"}})["select"] == "all"
    assert fringe_config({"fringe": {"prefer_blank": False}})["select"] == "all"
    assert fringe_config({"fringe": {"prefer_blank": True}})["select"] == "blanks"


def test_select_blanks_science_all(tmp_path: Path):
    paths = [
        _touch_sci(tmp_path / "n.BlankField.z.fits", obj="BlankField", airmass=1.05, mjd=58000.1),
        _touch_sci(tmp_path / "n.BlankField.z.01.fits", obj="BlankField", airmass=1.06, mjd=58000.11),
        _touch_sci(tmp_path / "n.NGC.z.fits", obj="NGC_6791", airmass=1.10, mjd=58000.12),
        _touch_sci(tmp_path / "n.SA.z.fits", obj="SA_103", airmass=1.12, mjd=58000.13),
        _touch_sci(tmp_path / "n.NGC.z.01.fits", obj="NGC_6791", airmass=2.50, mjd=58000.50),
    ]

    blanks = select_fringe_inputs(
        paths,
        select="blanks",
        min_frames=2,
        blanks_fallback="none",
        sky_tol_frac=0,
        min_sky=0,
        min_exptime=0,
        interactive=False,
    )
    assert len(blanks) == 2
    assert all("Blank" in p.name for p in blanks)

    sci = select_fringe_inputs(
        paths,
        select="science",
        airmass_tol=0.2,
        time_tol_hours=0.0,
        sky_tol_frac=0,
        min_sky=0,
        min_exptime=0,
        interactive=False,
    )
    # Median AM among science ≈ 1.12; high-airmass frame (2.50) drops
    assert len(sci) == 2
    assert all(p.name != "n.NGC.z.01.fits" for p in sci)
    assert all("Blank" not in p.name for p in sci)

    all_sim = select_fringe_inputs(
        paths,
        select="all",
        airmass_tol=0.2,
        time_tol_hours=6.0,
        min_frames=2,
        sky_tol_frac=0,
        min_sky=0,
        min_exptime=0,
        interactive=False,
    )
    # high-airmass frame should drop
    assert all(p.name != "n.NGC.z.01.fits" for p in all_sim)
    assert len(all_sim) >= 2


def test_filter_similar_airmass(tmp_path: Path):
    paths = [
        _touch_sci(tmp_path / "a.Obj.z.fits", obj="A", airmass=1.1, mjd=58000.0),
        _touch_sci(tmp_path / "b.Obj.z.fits", obj="B", airmass=1.15, mjd=58000.01),
        _touch_sci(tmp_path / "c.Obj.z.fits", obj="C", airmass=2.0, mjd=58000.02),
    ]
    kept = filter_similar_conditions(
        paths, airmass_tol=0.2, time_tol_hours=0, sky_tol_frac=0, min_sky=0, min_exptime=0
    )
    assert len(kept) == 2


def test_hall_date_plus_ut_hours_sets_mjd():
    hdr = fits.Header({"DATE-OBS": "2026-05-16", "UT": 7.345647})
    mjd = _frame_mjd(hdr)
    assert mjd is not None
    # 7.345647 h after 2026-05-16 00:00
    assert abs((mjd - int(mjd)) * 24 - 7.345647) < 1e-6
    later = _frame_mjd(fits.Header({"DATE-OBS": "2026-05-16", "UT": 9.0}))
    assert later is not None and later - mjd > 1.5 / 24


def test_fringe_catalog_lists_time_and_coordinates(tmp_path: Path):
    paths = [
        _write_sci(
            tmp_path / "n.Archinal-1.z.01.fits",
            obj="Archinal-1",
            AIRMASS=1.77,
            EXPTIME=4.0,
            **{"DATE-OBS": "2026-05-16", "UT": 7.345647, "RA": "18:54:49.9", "DEC": "+05:32:54"},
            sky=410.0,
        ),
        _write_sci(
            tmp_path / "n.Blank.z.fits",
            obj="Blank",
            AIRMASS=1.12,
            EXPTIME=300.0,
            **{"DATE-OBS": "2026-05-16", "UT": 4.0, "RA": "12:00:00.0", "DEC": "+20:00:00"},
            sky=380.0,
        ),
    ]
    rows = collect_fringe_catalog(paths)
    table = format_fringe_catalog(rows, [paths[1]])
    assert "18:54:49.9" in table
    assert "+05:32:54" in table
    assert "07:20:44" in table
    assert "12:00:00.0" in table
    assert "Archinal-1" in table
    # second row is auto-kept
    assert any(line.strip().startswith("2") and "*" in line for line in table.splitlines())


def test_offer_fringe_frames_pick_from_list(tmp_path: Path, monkeypatch):
    paths = [
        _write_sci(
            tmp_path / "n.A.z.fits",
            obj="A",
            AIRMASS=1.1,
            **{"DATE-OBS": "2026-05-16", "UT": 3.0, "RA": "01:00:00.0", "DEC": "+10:00:00"},
        ),
        _write_sci(
            tmp_path / "n.B.z.fits",
            obj="B",
            AIRMASS=1.12,
            **{"DATE-OBS": "2026-05-16", "UT": 3.1, "RA": "02:00:00.0", "DEC": "+11:00:00"},
        ),
        _write_sci(
            tmp_path / "n.C.z.fits",
            obj="C",
            AIRMASS=2.0,
            **{"DATE-OBS": "2026-05-16", "UT": 6.0, "RA": "03:00:00.0", "DEC": "+12:00:00"},
        ),
    ]
    rows = collect_fringe_catalog(paths)
    auto = [paths[0], paths[1]]
    answers = iter(["2", "1 3"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    chosen = offer_fringe_frames(
        rows,
        auto,
        cuts={"airmass_tol": 0.15, "time_tol_hours": 1.5, "sky_tol_frac": 0, "min_sky": 0, "min_exptime": 0},
        min_frames=2,
        interactive=True,
        label="z",
    )
    assert [p.name for p in chosen] == ["n.A.z.fits", "n.C.z.fits"]


def test_choose_enter_uses_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    assert choose("Pick:", [("auto", "Auto"), ("pick", "Pick")], default="auto") == "auto"


def test_ask_index_picks_default_and_range(monkeypatch):
    answers = iter(["", "2-4", "all"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert ask_index_picks(5, default=[1, 3]) == [1, 3]
    assert ask_index_picks(5) == [2, 3, 4]
    assert ask_index_picks(3) == [1, 2, 3]
