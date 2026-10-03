"""End-to-end pipeline: ``python -m src.pipeline [--input FILE] [--synthetic]``.

Stages: data -> features -> train -> explain -> segment -> score -> report. Every stage reads/writes files, so a
stage can be re-run alone with ``--stages``. All randomness is seeded from config.yaml.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.clean import clean_transactions
from src.data.load import load_raw
from src.data.validate import validate_features, validate_transactions
from src.explainability import explain as ex
from src.features.builders import feature_columns, make_train_and_scoring_tables
from src.features.sequences import build_sequences, category_vocab
from src.features.transform import FeatureTransformer, select_features_by_vif
from src.models.batch_score import score_snapshot
from src.models.common import make_splits
from src.models.predictor import Predictor
from src.models.train_classical import (
    DISPLAY_NAMES,
    train_all_classical,
    train_clv_models,
    train_next_category,
)
from src.models.train_dl import train_ann, train_autoencoder, train_lstm
from src.segmentation.cluster import run_segmentation
from src.utils.config import artifacts_path, load_config, path_of
from src.utils.experiment_log import ExperimentLogger, read_log, reset_log
from src.utils.logging_utils import get_logger

logger = get_logger("pipeline")
STAGES = ["data", "features", "train", "explain", "segment", "score", "report"]


def _meta_path(cfg: dict[str, Any]) -> Any:
    return artifacts_path(cfg, "metadata.json")


def _load_meta(cfg: dict[str, Any]) -> dict[str, Any]:
    p = _meta_path(cfg)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_meta(cfg: dict[str, Any], **updates: Any) -> dict[str, Any]:
    meta = _load_meta(cfg)
    meta.update(updates)
    _meta_path(cfg).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def _json(cfg: dict[str, Any], name: str, obj: Any) -> None:
    artifacts_path(cfg, "metrics", name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ----------------------------------------------------------------------------------------------- stages
def stage_data(cfg: dict[str, Any], input_path: str | None) -> None:
    """Load, validate, clean and persist transactions."""
    raw = load_raw(cfg, input_path)
    raw_val = validate_transactions(raw, cfg)
    tx, report = clean_transactions(raw, cfg)
    clean_val = validate_transactions(tx, cfg)
    report["raw_validation"], report["clean_validation"] = raw_val, clean_val
    out = path_of(cfg, "clean_transactions")
    out.parent.mkdir(parents=True, exist_ok=True)
    tx.to_csv(out, index=False)
    path_of(cfg, "cleaning_report").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def read_clean(cfg: dict[str, Any]) -> pd.DataFrame:
    """Read the cleaned transaction file with correct dtypes."""
    tx = pd.read_csv(path_of(cfg, "clean_transactions"), parse_dates=["invoice_date"], low_memory=False)
    tx["customer_id"] = tx["customer_id"].astype("Int64")
    for col in ("is_return", "is_price_adjustment", "is_admin_code", "is_entry_error", "use_for_customer_model",
                "has_customer_id", "is_outlier", "is_bulk_outlier"):
        tx[col] = tx[col].astype(bool)
    tx["description"] = tx["description"].astype("string")
    return tx


def stage_features(cfg: dict[str, Any]) -> None:
    """Point-in-time feature tables (training snapshot with labels, scoring snapshot without)."""
    tx = read_clean(cfg)
    train_tbl, score_tbl, stock_freq, cuts = make_train_and_scoring_tables(tx, cfg)
    ft = FeatureTransformer(cfg).fit(train_tbl)
    train_tbl["engagement_score"] = ft.engagement_score(train_tbl)
    score_tbl["engagement_score"] = ft.engagement_score(score_tbl)
    validate_features(train_tbl, feature_columns(cfg))
    validate_features(score_tbl, feature_columns(cfg))

    X_all = ft.transform(train_tbl)
    kept, vif = select_features_by_vif(X_all, cfg["features"]["vif_threshold"], cfg["features"]["vif_protected"])
    dropped = [c for c in X_all.columns if c not in kept]
    logger.info("VIF pruning dropped: %s", dropped)

    pdir = path_of(cfg, "processed_dir")
    pdir.mkdir(parents=True, exist_ok=True)
    train_tbl.to_csv(path_of(cfg, "features_train"), index=False)
    score_tbl.to_csv(path_of(cfg, "features_scoring"), index=False)
    vif.to_csv(pdir / "vif_report.csv", index=False)
    joblib.dump(ft, artifacts_path(cfg, "transformer.joblib"))
    joblib.dump(stock_freq, artifacts_path(cfg, "stockcode_frequency.joblib"))
    _save_meta(cfg, cutoffs=cuts, model_features=kept, vif_dropped=dropped,
               churn_rate_train_snapshot=float(train_tbl["churn"].mean()), n_customers_train=int(len(train_tbl)),
               n_customers_scoring=int(len(score_tbl)))


def _split_xy(X: Any, y: np.ndarray, sp: dict[str, np.ndarray], numpy: bool = False) -> dict[str, tuple[Any, np.ndarray]]:
    return {k: ((X.iloc[v].to_numpy() if numpy else X.iloc[v]), y[v]) for k, v in sp.items()}


def comparison_table(cfg: dict[str, Any], run_id: str) -> pd.DataFrame:
    """Final churn model comparison, regenerated from the experiment log (ranked by ROC-AUC then recall)."""
    log = read_log(cfg, run_id)
    t = log[(log["task"] == "churn") & (log["split"] == "test")].copy()
    cols = {"model": "model", "metrics.cv_roc_auc_mean": "cv_auc_mean", "metrics.cv_roc_auc_std": "cv_auc_std",
            "metrics.accuracy": "accuracy", "metrics.precision": "precision", "metrics.recall": "recall",
            "metrics.f1": "f1", "metrics.roc_auc": "roc_auc", "metrics.threshold": "threshold", "train_time_sec": "train_time_sec"}
    t = t[list(cols)].rename(columns=cols)
    t["model_name"] = t["model"].map(DISPLAY_NAMES)
    t = t.sort_values(["roc_auc", "recall"], ascending=False).reset_index(drop=True)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    return t


def stage_train(cfg: dict[str, Any], run_id: str) -> None:
    """Train and evaluate every model; select and persist the production churn model."""
    reset_log(cfg)
    exp = ExperimentLogger(cfg, run_id)
    tx = read_clean(cfg)
    meta = _load_meta(cfg)
    train_tbl = pd.read_csv(path_of(cfg, "features_train"), parse_dates=["first_purchase", "last_purchase"])
    ft: FeatureTransformer = joblib.load(artifacts_path(cfg, "transformer.joblib"))
    X = ft.transform(train_tbl)[meta["model_features"]]
    y = train_tbl["churn"].to_numpy()
    sp = make_splits(y, cfg)
    _json(cfg, "split_sizes.json", {k: int(len(v)) for k, v in sp.items()} | {"churn_rate": float(y.mean())})
    joblib.dump({k: train_tbl["customer_id"].to_numpy()[v] for k, v in sp.items()}, artifacts_path(cfg, "split_ids.joblib"))

    clas = train_all_classical(_split_xy(X, y, sp), cfg, exp)
    clv_y = train_tbl["clv_target"].to_numpy()
    clv = train_clv_models(_split_xy(X, clv_y, sp), cfg, exp)
    _json(cfg, "clv_metrics.json", {k: ({"val": v["val"], "test": v["test"], "params": v["params"], "cv_r2": v["cv_r2"]}
                                         if k != "selected" else v) for k, v in clv.items()})
    nxt = train_next_category(X, train_tbl["next_category"], sp, cfg, exp)
    _json(cfg, "next_category.json", nxt["test"])

    ann = train_ann(_split_xy(X, y, sp, numpy=True), cfg, exp)
    cut = pd.Timestamp(meta["cutoffs"]["train_cutoff"])
    num, cat = build_sequences(tx, train_tbl["customer_id"].to_numpy(), cut, cfg)
    seq = {k: {"num": num[v], "cat": cat[v]} for k, v in sp.items()}
    lstm = train_lstm(seq, {k: y[v] for k, v in sp.items()}, len(category_vocab(cfg)), cfg, exp)
    train_autoencoder({k: train_tbl.iloc[v] for k, v in sp.items()}, cfg, exp)

    cmp_tbl = comparison_table(cfg, run_id)
    cmp_tbl.to_csv(artifacts_path(cfg, "metrics", "model_comparison.csv"), index=False)

    cands = [m for m in cfg["models"]["production_candidates"] if m in clas]
    prod = max(cands, key=lambda m: (clas[m]["val"]["roc_auc"], clas[m]["val"]["recall"]))
    joblib.dump(clas[prod]["estimator"], artifacts_path(cfg, "production_model.joblib"))
    bg = X.iloc[sp["train"]].sample(min(cfg["explainability"]["background_size"], len(sp["train"])),
                                    random_state=cfg["project"]["seed"])
    joblib.dump(bg, artifacts_path(cfg, "explain_background.joblib"))
    _save_meta(cfg, run_id=run_id, production_model=prod, churn_threshold=clas[prod]["threshold"],
               production_cv_auc=clas[prod]["cv_auc_mean"], feature_medians=X.median().to_dict(),
               seed=cfg["project"]["seed"], ann_test_auc=ann["test"]["roc_auc"], lstm_test_auc=lstm["test"]["roc_auc"],
               versions={"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
               trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def stage_explain(cfg: dict[str, Any]) -> None:
    """SHAP (global/local), LIME comparison, partial dependence."""
    meta = _load_meta(cfg)
    train_tbl = pd.read_csv(path_of(cfg, "features_train"), parse_dates=["first_purchase", "last_purchase"])
    ft: FeatureTransformer = joblib.load(artifacts_path(cfg, "transformer.joblib"))
    X = ft.transform(train_tbl)[meta["model_features"]]
    ids = joblib.load(artifacts_path(cfg, "split_ids.joblib"))
    Xt = X[train_tbl["customer_id"].isin(ids["test"]).to_numpy()].reset_index(drop=True)
    model = joblib.load(artifacts_path(cfg, "production_model.joblib"))
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    thr = meta["churn_threshold"]
    fdir = ex.figures_dir(cfg)

    explainer = ex.make_explainer(clf)
    sv, base = ex.shap_matrix(explainer, Xt)
    imp = ex.global_importance(sv, list(Xt.columns))
    imp.to_csv(artifacts_path(cfg, "metrics", "shap_global_importance.csv"), index=False)
    ex.save_summary_plot(sv, Xt, fdir / "shap_summary.png")

    proba = model.predict_proba(Xt)[:, 1]
    reps = ex.pick_representative(proba, thr)
    for name, i in reps.items():
        ex.save_force_plot(base, sv[i], Xt.iloc[i], f"{name.replace('_', ' ')} customer (P(churn)={proba[i]:.2f})",
                           fdir / f"shap_force_{name}.png")
    top = list(imp["feature"].head(cfg["explainability"]["pdp_top_features"]))
    ex.save_pdp(clf, Xt, top, fdir / "pdp_top_features.png")

    bg = joblib.load(artifacts_path(cfg, "explain_background.joblib"))
    i = reps["borderline"]
    lime_pairs = ex.lime_explain(clf, bg, Xt.iloc[i], cfg)
    cmpd = ex.compare_lime_shap(lime_pairs, sv[i], list(Xt.columns))
    ex.save_lime_vs_shap_plot(lime_pairs, sv[i], list(Xt.columns), fdir / "lime_vs_shap.png")
    text = ex.explain_in_words(dict(zip(Xt.columns, sv[i])), Xt.iloc[i].to_dict(), meta["feature_medians"], proba[i], thr,
                               cfg["explainability"]["top_reasons"])
    _json(cfg, "explain_report.json", {"top_features": top, "lime_vs_shap": cmpd, "example_explanation": text,
                                       "representative_customers": reps})
    _save_meta(cfg, global_importance=imp.head(15).to_dict("records"))


def stage_segment(cfg: dict[str, Any]) -> None:
    """Segment the latest snapshot."""
    feats = pd.read_csv(path_of(cfg, "features_scoring"))
    res = run_segmentation(feats, cfg)
    pdir = path_of(cfg, "processed_dir")
    res["assignments"].to_csv(pdir / "segments.csv", index=False)
    res["profiles"].to_csv(artifacts_path(cfg, "metrics", "segment_profiles.csv"), index=False)
    _json(cfg, "segmentation_metrics.json", res["metrics"])


def stage_score(cfg: dict[str, Any]) -> None:
    """Score the latest snapshot and add predicted churn to the segment profiles."""
    preds = score_snapshot(cfg, Predictor(cfg))
    pdir = path_of(cfg, "processed_dir")
    preds.to_csv(pdir / "predictions.csv", index=False)
    seg = pd.read_csv(pdir / "segments.csv")
    prof = pd.read_csv(artifacts_path(cfg, "metrics", "segment_profiles.csv"))
    m = preds.merge(seg, on="customer_id").groupby("kmeans_segment").agg(
        mean_churn_probability=("churn_probability", "mean"), high_risk_pct=("risk_tier", lambda s: 100 * (s == "High").mean()))
    prof = prof.merge(m.round(3), left_on="segment_id", right_index=True, how="left")
    prof.to_csv(artifacts_path(cfg, "metrics", "segment_profiles.csv"), index=False)
    cmp_tbl = pd.read_csv(artifacts_path(cfg, "metrics", "model_comparison.csv"))
    prod_row = cmp_tbl[cmp_tbl["model"] == _load_meta(cfg)["production_model"]].iloc[0]
    _save_meta(cfg, production_test_metrics={k: float(prod_row[k]) for k in
                                             ["roc_auc", "recall", "precision", "f1", "accuracy"]})
    _save_meta(cfg, scored_customers=int(len(preds)), high_risk_customers=int((preds["risk_tier"] == "High").sum()),
               anomalies_flagged=int(preds["is_anomaly"].sum()))


def stage_report(cfg: dict[str, Any], database_url: str | None) -> None:
    """Diagrams, benchmarks and the PDF report."""
    from src.utils.benchmark import run_benchmarks
    from src.utils.diagrams import make_diagrams
    from src.utils.report import build_report

    make_diagrams(cfg)
    run_benchmarks(cfg)
    build_report(cfg)


def run_pipeline(cfg: dict[str, Any], input_path: str | None = None, synthetic: bool = False,
                 stages: list[str] | None = None) -> None:
    """Run the requested stages in order."""
    stages = stages or STAGES
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    if "data" in stages:
        _save_meta(cfg, data_source="SYNTHETIC STAND-IN (not real data)" if synthetic else cfg["project"]["data_label"],
                   synthetic=synthetic)
    for st in stages:
        logger.info("=== stage: %s ===", st)
        if st == "data":
            stage_data(cfg, input_path)
        elif st == "features":
            stage_features(cfg)
        elif st == "train":
            stage_train(cfg, run_id)
        elif st == "explain":
            stage_explain(cfg)
        elif st == "segment":
            stage_segment(cfg)
        elif st == "score":
            stage_score(cfg)
        elif st == "report":
            stage_report(cfg, None)
        else:
            raise ValueError(f"unknown stage {st}")


def main(argv: list[str] | None = None) -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description="Run the customer-behavior-prediction pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--input", default=None, help="raw CSV/XLSX path (skips ucimlrepo)")
    ap.add_argument("--synthetic", action="store_true", help="mark outputs as synthetic stand-in data")
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    args = ap.parse_args(argv)
    run_pipeline(load_config(args.config), args.input, args.synthetic, args.stages)


if __name__ == "__main__":
    sys.exit(main())
