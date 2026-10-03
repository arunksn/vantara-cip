"""Shared fixtures. The session fixture runs the real pipeline on a SMALL synthetic dataset in a temp directory."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, override_config  # noqa: E402

FAST = {
    "cv": {"folds": 2, "n_iter": 1},
    "models": {"random_forest": {"n_estimators": [30], "max_depth": [6], "min_samples_leaf": [5]},
               "xgboost": {"n_estimators": 60}, "lightgbm": {"n_estimators": 60},
               "next_category": {"n_estimators": 30}},
    "deep_learning": {"ann": {"epochs": 3}, "lstm": {"epochs": 3}, "autoencoder": {"epochs": 3}, "cv_epochs_cap": 2},
    "explainability": {"lime_num_samples": 200, "background_size": 100},
    "segmentation": {"k_min": 3, "k_max": 4},
    "serving": {"api_benchmark_requests": 10},
}


@pytest.fixture(scope="session")
def base_cfg() -> dict:
    return load_config(ROOT / "config" / "config.yaml")


@pytest.fixture(scope="session")
def trained_env(tmp_path_factory, base_cfg):
    """Run data -> score on ~800 synthetic customers; returns (cfg, tmp_root)."""
    import os

    from src.pipeline import run_pipeline
    from tests.synthetic import make_synthetic_transactions

    tmp = tmp_path_factory.mktemp("proj")
    cfg = override_config(base_cfg, FAST)
    cfg["_root"] = str(tmp)
    cfg_file = tmp / "config.yaml"
    out = {k: v for k, v in cfg.items() if k != "_root"}
    cfg_file.write_text(yaml.safe_dump(out), encoding="utf-8")
    raw = tmp / "raw.csv.gz"
    make_synthetic_transactions(800, seed=3).to_csv(raw, index=False)
    os.environ["PROJECT_ROOT"] = str(tmp)
    os.environ["CONFIG_PATH"] = str(cfg_file)
    (tmp / "frontend").mkdir()
    (tmp / "frontend" / "dashboard.py").write_text((ROOT / "frontend" / "dashboard.py").read_text(encoding="utf-8"), encoding="utf-8")
    run_pipeline(cfg, str(raw), synthetic=True, stages=["data", "features", "train", "explain", "segment", "score"])
    yield cfg, tmp
    os.environ.pop("PROJECT_ROOT", None)
    os.environ.pop("CONFIG_PATH", None)
