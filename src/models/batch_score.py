"""Batch scoring job: score the latest customer snapshot and (optionally) load PostgreSQL.

Usage:
    python -m src.models.batch_score              # writes data/processed/predictions.csv, segments.csv
    python -m src.models.batch_score --load-db    # also (re)loads the database from DATABASE_URL
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.features.builders import feature_columns
from src.models.predictor import Predictor
from src.utils.config import load_config, path_of
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def score_snapshot(cfg: dict[str, Any], predictor: Predictor | None = None) -> pd.DataFrame:
    """Score the scoring snapshot (features as of the end of the data) with SHAP reasons."""
    feats = pd.read_csv(path_of(cfg, "features_scoring"), parse_dates=["first_purchase", "last_purchase"])
    pred = predictor or Predictor(cfg)
    out = pred.predict(feats, with_shap=True)
    out.insert(0, "customer_id", feats["customer_id"].to_numpy())
    out["at_risk_value"] = out["churn_probability"] * feats["historical_clv"].to_numpy()
    out["recommended_categories"] = out["recommended_categories"].map(lambda v: ",".join(v))
    out["model_version"] = pred.meta["run_id"]
    out["scored_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
    return out


def customers_frame(cfg: dict[str, Any]) -> pd.DataFrame:
    """Customers table content from the scoring snapshot."""
    feats = pd.read_csv(path_of(cfg, "features_scoring"), parse_dates=["first_purchase", "last_purchase"])
    cols = feature_columns(cfg)
    feats["features_json"] = feats[cols].apply(lambda r: json.dumps({k: float(v) for k, v in r.items()}), axis=1)
    keep = ["customer_id", "country", "first_purchase", "last_purchase", "recency_days", "frequency",
            "total_spend", "historical_clv", "engagement_score", "features_json"]
    return feats[keep]


def load_database(cfg: dict[str, Any]) -> dict[str, int]:
    """(Re)load all four tables from the processed files."""
    from src.utils.db import get_engine, load_all

    cust = customers_frame(cfg)
    preds = pd.read_csv(path_of(cfg, "processed_dir") / "predictions.csv", parse_dates=["scored_at"])
    segs = pd.read_csv(path_of(cfg, "processed_dir") / "segments.csv")
    tx_path = path_of(cfg, "clean_transactions")
    tx = None
    if tx_path.exists():
        raw = pd.read_csv(tx_path, parse_dates=["invoice_date"], low_memory=False)
        raw = raw[raw["use_for_customer_model"] & raw["customer_id"].isin(cust["customer_id"])].copy()
        raw["description"] = raw["description"].astype(str).str.slice(0, 255)
        raw["customer_id"] = raw["customer_id"].astype("int64")
        tx = raw[["invoice", "stock_code", "description", "category", "quantity", "invoice_date", "unit_price",
                  "line_total", "is_return", "customer_id", "country"]]
    return load_all(get_engine(), cust, tx, preds, segs)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--load-db", action="store_true", help="reload PostgreSQL from processed files")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pdir = path_of(cfg, "processed_dir")
    if not (pdir / "predictions.csv").exists():
        preds = score_snapshot(cfg)
        preds.to_csv(pdir / "predictions.csv", index=False)
    if args.load_db:
        logger.info("database load counts: %s", load_database(cfg))


if __name__ == "__main__":
    main()
