"""Classical ML models for churn (LR, DT, RF, XGBoost, LightGBM, KNN) and CLV regression.

Discipline (PRD 8.3): fixed seed, stratified split, hyper-parameter search with 5-fold stratified CV on the TRAINING
set only, early stopping on the validation split, class imbalance handled on training data only, test set used once.
"""
from __future__ import annotations

import time
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

from src.models.common import classification_metrics, regression_metrics, select_threshold
from src.utils.config import artifacts_path
from src.utils.experiment_log import ExperimentLogger
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

CLASSICAL_MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm", "knn"]
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree",
    "random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM", "knn": "KNN",
    "ann": "ANN (feed-forward)", "lstm": "LSTM (sequence)",
}


def _grid_size(space: dict[str, list[Any]]) -> int:
    n = 1
    for v in space.values():
        n *= len(v)
    return n


def build_pipeline(name: str, cfg: dict[str, Any], y_train: np.ndarray) -> tuple[Pipeline, dict[str, list[Any]], bool]:
    """Return (pipeline with a final 'clf' step, search space with clf__ prefixes, uses_eval_set)."""
    seed = cfg["project"]["seed"]
    mcfg = cfg["models"]
    smote = cfg["imbalance"]["method"] == "smote"
    n_pos = max(int(y_train.sum()), 1)
    spw = 1.0 if smote else (len(y_train) - n_pos) / n_pos
    cw = None if smote else "balanced"
    steps: list[tuple[str, Any]] = []
    uses_eval = False

    if name == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
        clf = LogisticRegression(solver="liblinear", max_iter=2000, class_weight=cw, random_state=seed)
        space = {"C": mcfg[name]["C"], "l1_ratio": mcfg[name]["l1_ratio"]}
    elif name == "decision_tree":
        clf = DecisionTreeClassifier(class_weight=cw, random_state=seed)
        space = {"max_depth": mcfg[name]["max_depth"], "min_samples_leaf": mcfg[name]["min_samples_leaf"]}
    elif name == "random_forest":
        clf = RandomForestClassifier(class_weight=None if smote else "balanced_subsample", n_jobs=1, random_state=seed)
        space = {k: mcfg[name][k] for k in ("n_estimators", "max_depth", "min_samples_leaf")}
    elif name == "xgboost":
        clf = XGBClassifier(n_estimators=mcfg[name]["n_estimators"], early_stopping_rounds=mcfg[name]["early_stopping_rounds"],
                            eval_metric="auc", tree_method="hist", scale_pos_weight=spw, random_state=seed,
                            n_jobs=1, verbosity=0)
        space = {k: mcfg[name][k] for k in ("learning_rate", "max_depth", "subsample", "colsample_bytree")}
        uses_eval = True
    elif name == "lightgbm":
        clf = LGBMClassifier(n_estimators=mcfg[name]["n_estimators"], class_weight=cw, random_state=seed,
                             n_jobs=1, verbosity=-1)
        space = {k: mcfg[name][k] for k in ("learning_rate", "num_leaves", "subsample", "colsample_bytree")}
        uses_eval = True
    elif name == "knn":
        steps.append(("scaler", StandardScaler()))
        clf = KNeighborsClassifier(n_jobs=1)
        space = {"n_neighbors": mcfg[name]["n_neighbors"], "weights": mcfg[name]["weights"]}
    else:
        raise ValueError(f"unknown model {name}")
    if smote:
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("clf", clf))
    return Pipeline(steps), {f"clf__{k}": v for k, v in space.items()}, uses_eval


def train_classifier(
    name: str, splits_xy: dict[str, tuple[np.ndarray, np.ndarray]], cfg: dict[str, Any], exp: ExperimentLogger
) -> dict[str, Any]:
    """Tune, fit and evaluate one churn classifier. Test metrics are computed exactly once, at the end."""
    seed = cfg["project"]["seed"]
    X_tr, y_tr = splits_xy["train"]
    X_va, y_va = splits_xy["val"]
    X_te, y_te = splits_xy["test"]
    pipe, space, uses_eval = build_pipeline(name, cfg, y_tr)
    cv = StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=seed)
    search = RandomizedSearchCV(pipe, space, n_iter=min(cfg["cv"]["n_iter"], _grid_size(space)), cv=cv,
                                scoring=cfg["cv"]["scoring"], random_state=seed, n_jobs=1, refit=True)
    fit_params: dict[str, Any] = {}
    if uses_eval:
        if name == "xgboost":
            fit_params["clf__eval_set"] = [(X_va, y_va)]
            fit_params["clf__verbose"] = False
        else:
            fit_params["clf__eval_X"] = X_va
            fit_params["clf__eval_y"] = y_va
            fit_params["clf__callbacks"] = [lgb.early_stopping(cfg["models"][name]["early_stopping_rounds"], verbose=False)]
            fit_params["clf__eval_metric"] = "auc"
    t0 = time.time()
    search.fit(X_tr, y_tr, **fit_params)
    train_time = time.time() - t0

    best = search.best_estimator_
    p_va = best.predict_proba(X_va)[:, 1]
    thr = select_threshold(y_va, p_va, cfg["imbalance"]["min_recall_target"])
    p_te = best.predict_proba(X_te)[:, 1]
    val_m = classification_metrics(y_va, p_va, thr)
    test_m = classification_metrics(y_te, p_te, thr)
    i = search.best_index_
    cv_mean, cv_std = float(search.cv_results_["mean_test_score"][i]), float(search.cv_results_["std_test_score"][i])
    params = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    exp.log("churn", name, "val", {**val_m, "cv_roc_auc_mean": cv_mean, "cv_roc_auc_std": cv_std}, params, train_time)
    exp.log("churn", name, "test", {**test_m, "cv_roc_auc_mean": cv_mean, "cv_roc_auc_std": cv_std}, params, train_time)
    logger.info("trained %s: cv_auc=%.3f val_auc=%.3f test_auc=%.3f test_recall=%.3f", name, cv_mean,
                val_m["roc_auc"], test_m["roc_auc"], test_m["recall"])
    joblib.dump(best, artifacts_path(cfg, "classical", f"{name}.joblib"))
    return {"name": name, "estimator": best, "params": params, "threshold": thr, "val": val_m, "test": test_m,
            "cv_auc_mean": cv_mean, "cv_auc_std": cv_std, "p_val": p_va, "p_test": p_te, "train_time": train_time}


def train_all_classical(
    splits_xy: dict[str, tuple[np.ndarray, np.ndarray]], cfg: dict[str, Any], exp: ExperimentLogger,
    models: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Train every classical churn classifier."""
    return {m: train_classifier(m, splits_xy, cfg, exp) for m in (models or CLASSICAL_MODELS)}


def train_clv_models(
    splits_xy: dict[str, tuple[np.ndarray, np.ndarray]], cfg: dict[str, Any], exp: ExperimentLogger
) -> dict[str, Any]:
    """CLV regression: Ridge baseline and tuned XGBoost regressor; best by training-set CV R^2 is kept."""
    seed = cfg["project"]["seed"]
    ccfg = cfg["models"]["clv"]
    X_tr, y_tr = splits_xy["train"]
    X_va, y_va = splits_xy["val"]
    X_te, y_te = splits_xy["test"]
    cv = KFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=seed)
    candidates = {
        "ridge": (Pipeline([("scaler", StandardScaler()), ("clf", Ridge(random_state=seed))]),
                  {"clf__alpha": ccfg["ridge_alpha"]}),
        "xgb_regressor": (Pipeline([("clf", XGBRegressor(tree_method="hist", random_state=seed, n_jobs=1, verbosity=0))]),
                          {"clf__max_depth": ccfg["xgb_max_depth"], "clf__learning_rate": ccfg["xgb_learning_rate"],
                           "clf__n_estimators": ccfg["xgb_n_estimators"]}),
    }
    results: dict[str, Any] = {}
    for name, (pipe, space) in candidates.items():
        search = RandomizedSearchCV(pipe, space, n_iter=min(cfg["cv"]["n_iter"], _grid_size(space)), cv=cv,
                                    scoring="r2", random_state=seed, n_jobs=1)
        t0 = time.time()
        search.fit(X_tr, y_tr)
        tt = time.time() - t0
        best = search.best_estimator_
        val_m = regression_metrics(y_va, best.predict(X_va))
        test_m = regression_metrics(y_te, best.predict(X_te))
        i = search.best_index_
        cvm = {"cv_r2_mean": float(search.cv_results_["mean_test_score"][i]),
               "cv_r2_std": float(search.cv_results_["std_test_score"][i])}
        params = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
        exp.log("clv", name, "val", {**val_m, **cvm}, params, tt)
        exp.log("clv", name, "test", {**test_m, **cvm}, params, tt)
        results[name] = {"estimator": best, "val": val_m, "test": test_m, "params": params, "cv_r2": cvm["cv_r2_mean"]}
        logger.info("CLV %s: val_r2=%.3f test_r2=%.3f", name, val_m["r2"], test_m["r2"])
    # Selection uses 5-fold CV R^2 on the training set (more stable than one validation split); test is not used.
    best_name = max(results, key=lambda k: results[k]["cv_r2"])
    joblib.dump(results[best_name]["estimator"], artifacts_path(cfg, "classical", "clv_model.joblib"))
    results["selected"] = best_name
    return results


def train_next_category(
    X: pd.DataFrame, y: pd.Series, splits: dict[str, np.ndarray], cfg: dict[str, Any], exp: ExperimentLogger
) -> dict[str, Any]:
    """Multiclass next-purchase-category model (customers who purchased in the horizon window)."""
    from sklearn.ensemble import RandomForestClassifier as RF
    from sklearn.metrics import accuracy_score, f1_score

    seed = cfg["project"]["seed"]
    ncfg = cfg["models"]["next_category"]
    has = y.notna().to_numpy()
    tr = np.intersect1d(splits["train"], np.flatnonzero(has))
    te = np.intersect1d(splits["test"], np.flatnonzero(has))
    clf = RF(n_estimators=ncfg["n_estimators"], min_samples_leaf=ncfg["min_samples_leaf"], class_weight="balanced_subsample",
             n_jobs=1, random_state=seed)
    t0 = time.time()
    clf.fit(X.iloc[tr], y.iloc[tr])
    pred = clf.predict(X.iloc[te])
    m = {"accuracy": float(accuracy_score(y.iloc[te], pred)),
         "macro_f1": float(f1_score(y.iloc[te], pred, average="macro"))}
    exp.log("next_category", "random_forest", "test", m, {"n_estimators": ncfg["n_estimators"]}, time.time() - t0)
    joblib.dump(clf, artifacts_path(cfg, "classical", "next_category_model.joblib"))
    return {"estimator": clf, "test": m}
