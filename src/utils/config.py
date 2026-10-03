"""Configuration loader. All paths, hyperparameters and thresholds come from YAML."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("config") / "config.yaml"


def project_root() -> Path:
    """Return the repository root (override with the PROJECT_ROOT environment variable)."""
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the YAML config. Resolution order: argument, CONFIG_PATH env var, default file."""
    root = project_root()
    cfg_path = Path(path or os.environ.get("CONFIG_PATH", DEFAULT_CONFIG))
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with open(cfg_path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    cfg["_root"] = str(root)
    if os.environ.get("MODEL_ARTIFACTS_DIR"):
        cfg["paths"]["artifacts_dir"] = os.environ["MODEL_ARTIFACTS_DIR"]
    if os.environ.get("LOG_LEVEL"):
        cfg["logging"]["level"] = os.environ["LOG_LEVEL"]
    return cfg


def override_config(cfg: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of cfg with nested updates applied (used by tests)."""
    new = copy.deepcopy(cfg)

    def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                _merge(dst[key], value)
            else:
                dst[key] = value

    _merge(new, updates)
    return new


def path_of(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a path from cfg['paths'] against the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def artifacts_path(cfg: dict[str, Any], *parts: str) -> Path:
    """Return (and create the parent of) a path inside the model-artifacts directory."""
    p = path_of(cfg, "artifacts_dir").joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
