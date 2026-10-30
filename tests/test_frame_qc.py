"""Per-frame QC (imstat habit): flag odd frames, exclude them by file name."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ccd_pipeline.frame_qc import (
    check_master_noise,
    flag_frame_outliers,
    flat_sets_by_filter,
    format_frame_table,
    frame_statistics,
    group_key,
)
from ccd_pipeline.inventory import filter_excluded_rows, is_excluded_file
from ccd_pipeline.night_init import save_exclude_files


def _bias_row(name: str, level: float, noise: float = 7.0) -> dict:
    return {
        "file": name, "kind": "bias", "object": "zero", "filter": "", "exptime": 0.0,
        "excluded": False, "clipped_median": level, "clipped_std": noise, "frac_sat": 0.0,
        "median": level, "clipped_mean": level, "mad_std": noise, "min": level - 30,
        "max": level + 3000, "n_hot": 5000,
    }


def _flat_row(name: str, filt: str, obj: str, level: float, std: float, sat: float = 0.0) -> dict:
    row = _bias_row(name, level, std)
    row.update({"kind": "flat", "filter": filt, "object": obj, "frac_sat": sat, "exptime": 5.0})
    return row


def test_frame_statistics_are_sigma_clipped():
    rng = np.random.default_rng(0)
    data = rng.normal(1600.0, 7.0, (200, 200))
    data[0, :50] = 60000.0  # cosmic rays / saturated pixels
    stats = frame_statistics(data)
    assert abs(stats["clipped_median"] - 1600.0) < 1.0
    assert abs(stats["clipped_std"] - 7.0) < 0.5
    assert stats["std"] > 100  # the plain std is ruined by the hot pixels
    assert stats["n_hot"] == 50 and stats["frac_sat"] == 50 / data.size


def test_bias_level_and_noise_outliers_are_flagged():
    rows = [_bias_row(f"b{i}.fits", 1615.0 + (i % 3) * 0.5) for i in range(10)]
    rows.append(_bias_row("low.fits", 1590.0))
    rows.append(_bias_row("noisy.fits", 1615.0, noise=12.0))
    cfg = {"qc": {}}
    flags = flag_frame_outliers(rows, "bias", cfg)
    assert set(flags) == {"low.fits", "noisy.fits"}
    assert "level" in flags["low.fits"][0] and "noise" in flags["noisy.fits"][0]
    table = format_frame_table(rows, flags, kind="bias")
    assert "⚠" in table and "sigma_clipped_stats" in table


def test_flats_group_by_set_and_only_flag_saturation_structure_or_faintness():
    dome = [_flat_row(f"d{i}.fits", "g", "dflat-g", 30000.0 - 50 * i, 570.0) for i in range(6)]
    twi = [_flat_row(f"t{i}.fits", "g", "tflat-g", 40000.0 / (i + 1), 1000.0 / (i + 1)) for i in range(4)]
    twi.append(_flat_row("sat.fits", "g", "tflat-g", 50000.0, 900.0, sat=0.02))
    twi.append(_flat_row("faint.fits", "g", "tflat-g", 900.0, 20.0))
    dome.append(_flat_row("cloud.fits", "g", "dflat-g", 29000.0, 1500.0))
    rows = dome + twi
    assert group_key(rows[0], "flat") == "g dflat-g"
    sets = flat_sets_by_filter(rows)
    assert sets["g"] == {"dflat-g": 7, "tflat-g": 6}
    flags = flag_frame_outliers(rows, "flat", {"qc": {}})
    # fading twilight levels are not outliers; saturation, faintness, structure are
    assert set(flags) == {"sat.fits", "faint.fits", "cloud.fits"}
    assert "saturated" in flags["sat.fits"][0]
    assert "flat_min_adu" in flags["faint.fits"][0]
    assert "structure" in flags["cloud.fits"][0]


def test_exclude_files_is_saved_and_honoured(tmp_path: Path):
    config_path = tmp_path / "night.yaml"
    config_path.write_text("# header\nnight_id: x\noverscan:\n  median: true\n")
    cfg = {"night_id": "x", "overscan": {"median": True}, "_config_path": str(config_path)}
    merged = save_exclude_files(config_path, cfg, ["raw/oct12.2021.fits", "oct12.2360.fits"])
    assert merged == ["oct12.2021.fits", "oct12.2360.fits"]
    on_disk = yaml.safe_load(config_path.read_text())
    assert on_disk["exclude_files"] == merged
    assert config_path.read_text().startswith("# header")
    assert is_excluded_file("/any/where/oct12.2021.fits", cfg)
    rows = [
        {"file": "oct12.2021.fits", "object": "zero", "obstype": "ZERO"},
        {"file": "oct12.2022.fits", "object": "zero", "obstype": "ZERO"},
    ]
    kept = filter_excluded_rows(rows, cfg, warn=False)
    assert [r["file"] for r in kept] == ["oct12.2022.fits"]
    # merging keeps earlier names
    assert "oct12.2021.fits" in save_exclude_files(config_path, cfg, ["oct12.2001.fits"])


def test_choose_many_accepts_lists_ranges_and_all(monkeypatch, capsys):
    from ccd_pipeline.prompt import choose_many

    options = [(f"id{i}", f"option {i}") for i in range(1, 11)]
    answers = iter(["2 4 6 8 10", "2,4", "3-5", "all", "11", "0 2", "11 2", "7"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert choose_many("Pick:", options) == ["id2", "id4", "id6", "id8", "id10"]
    assert choose_many("Pick:", options) == ["id2", "id4"]
    assert choose_many("Pick:", options) == ["id3", "id4", "id5"]
    assert choose_many("Pick:", options) == [f"id{i}" for i in range(1, 11)]
    assert choose_many("Pick:", options) == []  # 11 = None
    # "0 2" and "11 2" are rejected, then "7" is accepted
    assert choose_many("Pick:", options) == ["id7"]
    assert "Invalid choice" in capsys.readouterr().out


def test_master_noise_rule():
    ok = check_master_noise(1.5, 7.4, 30)
    assert ok[0][0] == "ok" and "×0.20" in ok[0][1]
    warn = check_master_noise(5.0, 7.4, 30)  # far above the 1.35 floor
    assert warn[0][0] == "warn"
    bad = check_master_noise(8.0, 7.4, 30)
    assert bad[0][0] == "fail"
