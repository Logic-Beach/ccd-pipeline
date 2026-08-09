"""Path resolution for night configs and RAW/<night> layouts."""

from __future__ import annotations

from pathlib import Path

from ccd_pipeline.config import config_paths_root, load_config
from ccd_pipeline.night_init import resolve_night_paths, _rel_to_root


def test_config_paths_root_uses_parent_of_configs(tmp_path: Path):
    campaign = tmp_path / "2017_JUN_JAS"
    configs = campaign / "configs"
    configs.mkdir(parents=True)
    cfg_path = configs / "hdi_2017jun29.yaml"
    cfg_path.write_text("night_id: x\npaths: {}\n")
    assert config_paths_root(cfg_path) == campaign.resolve()


def test_load_config_resolves_raw_relative_to_campaign(tmp_path: Path):
    campaign = tmp_path / "2017_JUN_JAS"
    raw = campaign / "RAW" / "2017JUN29"
    raw.mkdir(parents=True)
    (raw / "bias.fits").write_text("x")
    configs = campaign / "configs"
    configs.mkdir()
    cfg_path = configs / "hdi_2017jun29.yaml"
    cfg_path.write_text(
        "night_id: 2017JUN29\n"
        "paths:\n"
        "  raw_dir: RAW/2017JUN29\n"
        "  output_dir: reduced/2017JUN29\n"
        "  masters_dir: reduced/2017JUN29/masters\n"
    )
    cfg = load_config(cfg_path)
    assert cfg["paths"]["raw_dir"] == raw.resolve()
    assert cfg["paths"]["output_dir"] == (campaign / "reduced" / "2017JUN29").resolve()


def test_resolve_night_paths_raw_campaign_layout(tmp_path: Path):
    campaign = tmp_path / "2017_JUN_JAS"
    raw_night = campaign / "RAW" / "2017JUN29"
    raw_night.mkdir(parents=True)
    (campaign / "REDUCED").mkdir()
    night_id, raw_dir, reduced, night_root = resolve_night_paths(
        raw_night, project_root=tmp_path / "ccd-pipeline"
    )
    assert night_id == "2017JUN29"
    assert raw_dir == raw_night.resolve()
    assert reduced == (campaign / "REDUCED" / "2017JUN29").resolve()
    assert night_root == campaign.resolve()


def test_rel_to_root_absolute_without_project_marker(tmp_path: Path):
    campaign = tmp_path / "campaign"
    campaign.mkdir(parents=True)
    raw = campaign / "RAW" / "n"
    raw.mkdir(parents=True)
    # No pyproject/.git under campaign → store absolute
    assert _rel_to_root(raw, campaign) == str(raw.resolve())


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        for name in ("t1", "t2", "t3", "t4"):
            (p / name).mkdir()
        test_config_paths_root_uses_parent_of_configs(p / "t1")
        test_load_config_resolves_raw_relative_to_campaign(p / "t2")
        test_resolve_night_paths_raw_campaign_layout(p / "t3")
        test_rel_to_root_absolute_without_project_marker(p / "t4")
    print("all path tests OK")
