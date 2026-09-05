"""Configuration loading with deterministic, explicit overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries without mutating either input."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the default configuration and optionally layer a user YAML file over it."""
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        defaults = yaml.safe_load(handle) or {}
    if path is None:
        return defaults

    with path.open("r", encoding="utf-8") as handle:
        override = yaml.safe_load(handle) or {}
    if not isinstance(override, dict):
        raise ValueError("Configuration override must be a YAML mapping.")
    return _deep_merge(defaults, override)


def artifact_root(config: dict[str, Any]) -> Path:
    """Resolve and create the configured root for reproducible runtime artefacts."""
    root = Path(config["project"]["artifact_dir"])
    root.mkdir(parents=True, exist_ok=True)
    return root
