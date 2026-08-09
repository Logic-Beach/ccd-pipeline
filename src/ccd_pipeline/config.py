"""Configuration loading helpers.

Night YAML files list instrument keywords and data paths. Paths may be absolute
(external drives are fine) or relative. Relative paths resolve against the
ccd-pipeline project root when that is discoverable; otherwise against the
parent of a ``configs/`` directory (data-campaign layouts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path) -> Path:
    """Walk parents until pyproject.toml / .git is found."""
    start = Path(start).expanduser().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return start


def has_project_marker(root: Path) -> bool:
    root = Path(root)
    return (root / "pyproject.toml").exists() or (root / ".git").exists()


def config_paths_root(config_path: Path) -> Path:
    """Directory that relative ``paths:`` entries in a night YAML are relative to.

    Prefer a real project root (pyproject.toml / .git). When the YAML lives in a
    standalone ``configs/`` folder next to data (no repo markers), use the parent
    of ``configs/`` so ``RAW/2017JUN29`` resolves to ``<campaign>/RAW/2017JUN29``
    rather than ``<campaign>/configs/RAW/2017JUN29``.
    """
    config_path = Path(config_path).expanduser().resolve()
    start = config_path.parent
    root = find_project_root(start)
    if has_project_marker(root):
        return root
    if start.name.lower() == "configs":
        return start.parent
    return start


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open() as fh:
        cfg = yaml.safe_load(fh)

    root = config_paths_root(path)
    paths = cfg.setdefault("paths", {})
    for key, value in list(paths.items()):
        p = Path(value)
        if not p.is_absolute():
            paths[key] = (root / p).resolve()
        else:
            paths[key] = p.resolve()
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["masters_dir"]).mkdir(parents=True, exist_ok=True)
