"""Helpers for loading repository YAML config files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml



def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top of {config_path}.")
    return data



def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Backward-compatible alias kept for call sites that still reference YAML explicitly."""
    return load_config(path)