"""Project paths and YAML config loading (no hardcoded environment-specific paths)."""

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Project root: BASKETMIND_ROOT if set, else the directory holding pyproject.toml."""
    if env := os.environ.get("BASKETMIND_ROOT"):
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml not found above this file; set BASKETMIND_ROOT")


def load_config(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load configs/<name>.yaml (e.g. load_config("training"))."""
    path = (config_dir or project_root() / "configs") / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
