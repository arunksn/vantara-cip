"""End-to-end pipeline smoke tests: artifacts, reproducibility, database, dashboard, report."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.batch_score import load_database
from src.models.common import make_splits
from src.utils.config import artifacts_path, path_of
from src.utils.db import get_engine, monthly_revenue, read_sql


def test_artifacts_and_metadata(trained_env):
    cfg, _ = trained_env
    meta = json.loads(artifacts_path(cfg, "metadata.json").read_text())
    assert meta["synthetic"] is True and "SYNTHETIC" in meta["data_source"]
    for p in ["transformer.joblib", "production_model.joblib", "classical/clv_model.joblib", "deep_learning/autoencoder.keras",
              "deep_learning/ann.keras", "deep_learning/lstm.keras", "segmentation/segmentation.joblib"]:
        assert artifacts_path(cfg, p).exists(), p


def test_comparison_table_covers_all_models(trained_env):
    cfg, _ = trained_env
    t = pd.read_csv(artifacts_path(cfg, "metrics", "model_comparison.csv"))
    assert set(t["model"]) == {"logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm", "knn", "ann", "lstm"}
    assert t["roc_auc"].is_monotonic_decreasing and (t["rank"] == np.arange(1, 9)).all()
    assert t[["accuracy", "precision", "recall", "f1", "roc_auc", "cv_auc_mean"]].notna().all().all()


def test_split_reproducible_and_no_overlap(trained_env, base_cfg):
    cfg, _ = trained_env
    ids = joblib.load(artifacts_path(cfg, "split_ids.joblib"))
    assert not set(ids["train"]) & set(ids["test"]) and not set(ids["val"]) & set(ids["test"])
    tbl = pd.read_csv(path_of(cfg, "features_train"))
    again = make_splits(tbl["churn"].to_numpy(), cfg)
    assert np.array_equal(tbl["customer_id"].to_numpy()[again["test"]], ids["test"])


def test_predictions_and_segments(trained_env):
    cfg, _ = trained_env
    p = pd.read_csv(path_of(cfg, "processed_dir") / "predictions.csv")
    s = pd.read_csv(path_of(cfg, "processed_dir") / "segments.csv")
    assert p["customer_id"].is_unique and set(p["customer_id"]) == set(s["customer_id"])
    assert p["churn_probability"].between(0, 1).all() and (p["predicted_clv_90d"] >= 0).all()
    assert p["top_reasons"].str.contains("Main drivers").all()
    assert s["segment_label"].nunique() >= 2 and set(s["value_tier"]) == {"Low", "Medium", "High"}


def test_database_load_and_dashboard_smoke(trained_env, monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    cfg, tmp = trained_env
    url = f"sqlite:///{tmp_path}/t.db"
    monkeypatch.setenv("DATABASE_URL", url)
    counts = load_database(cfg)
    assert counts["customers"] == counts["predictions"] == counts["segments"] and counts["transactions"] > 0
    eng = get_engine(url)
    assert read_sql(eng, "SELECT COUNT(*) AS n FROM predictions").iloc[0, 0] == counts["predictions"]
    assert len(monthly_revenue(eng)) > 12
    at = AppTest.from_file(str(Path(tmp) / "frontend" / "dashboard.py"), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5


def test_diagrams_and_report(trained_env):
    from src.utils.diagrams import make_diagrams
    from src.utils.report import build_report

    cfg, _ = trained_env
    make_diagrams(cfg)
    pdf = build_report(cfg)
    assert pdf.exists() and pdf.stat().st_size > 20_000
    assert (path_of(cfg, "docs_dir") / "er_diagram.png").exists()
