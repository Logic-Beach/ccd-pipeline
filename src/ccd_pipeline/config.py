"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path) -> Path:
    """Walk parents until pyproject.toml / .git is found."""
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return start


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open() as fh:
        cfg = yaml.safe_load(fh)

    # Resolve relative data paths against the project root (not configs/)
    root = find_project_root(path.parent)
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
