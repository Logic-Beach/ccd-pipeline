"""Path resolution for night configs and RAW/<night> layouts."""

from __future__ import annotations

from pathlib import Path

from ccd_pipeline.config import config_paths_root, load_config
from ccd_pipeline.night_init import (
    camera_label,
    config_mirror_paths,
    night_config_paths,
    resolve_night_paths,
    save_overscan_biassec,
    _rel_to_root,
)


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


def test_camera_label_uses_detector_when_instrument_is_a_placeholder():
    assert camera_label("test", "s2kb") == "s2kb"
    assert camera_label("hdi", "s2kb") == "hdi"
    assert camera_label("", "s2kb") == "s2kb"


def test_night_config_paths_writes_data_folder_and_repo(tmp_path: Path):
    campaign = tmp_path / "2013_OCT"
    night = campaign / "RAW" / "2013OCT12"
    night.mkdir(parents=True)
    repo = tmp_path / "ccd-pipeline"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='ccd'\n")
    paths = night_config_paths(
        night_root=campaign, project_root=repo, filename="test_2013oct12.yaml"
    )
    assert paths[0] == (campaign / "configs" / "test_2013oct12.yaml").resolve()
    assert paths[1] == (repo / "configs" / "test_2013oct12.yaml").resolve()


def test_night_config_paths_single_when_data_is_the_repo(tmp_path: Path):
    repo = tmp_path / "ccd-pipeline"
    (repo / "configs").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='ccd'\n")
    paths = night_config_paths(
        night_root=repo, project_root=repo, filename="hdi_night.yaml"
    )
    assert paths == [(repo / "configs" / "hdi_night.yaml").resolve()]


def test_save_overscan_updates_night_and_repo_copies(tmp_path: Path, monkeypatch):
    campaign = tmp_path / "2013_OCT"
    raw = campaign / "RAW" / "2013OCT12"
    raw.mkdir(parents=True)
    repo = tmp_path / "ccd-pipeline"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='ccd'\n")
    monkeypatch.chdir(repo)
    text = (
        "# Auto-generated\n"
        "night_id: 2013OCT12\n"
        "paths:\n"
        f"  raw_dir: {raw}\n"
        "overscan:\n"
        "  median: true\n"
    )
    repo_cfg = repo / "configs" / "test_2013oct12.yaml"
    repo_cfg.parent.mkdir()
    repo_cfg.write_text(text)
    cfg = {
        "paths": {"raw_dir": raw},
        "overscan": {"median": True},
    }
    save_overscan_biassec(repo_cfg, cfg, "[1036:1056,1:1024]")
    night_cfg = campaign / "configs" / "test_2013oct12.yaml"
    assert "1036:1056" in repo_cfg.read_text()
    assert "1036:1056" in night_cfg.read_text()
    assert night_cfg.resolve() in config_mirror_paths(repo_cfg, cfg)


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
