"""Shared modelling helpers: stratified splits, metrics, operating-threshold selection."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def make_splits(y: np.ndarray, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Stratified train/validation/test split (default 70/15/15) with a fixed seed.

    Returns positional indices into the arrays the caller holds.
    """
    seed = cfg["project"]["seed"]
    sp = cfg["split"]
    idx = np.arange(len(y))
    trainval, test = train_test_split(idx, test_size=sp["test"], stratify=y, random_state=seed)
    val_frac = sp["val"] / (sp["train"] + sp["val"])
    train, val = train_test_split(trainval, test_size=val_frac, stratify=y[trainval], random_state=seed)
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    """Accuracy, precision, recall, F1, ROC-AUC and confusion matrix (positive class = churn)."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_threshold(y_val: np.ndarray, p_val: np.ndarray, min_recall: float) -> float:
    """Pick the validation-set operating threshold.

    Highest-F1 threshold among those reaching ``min_recall`` on the churned class; if none reaches it,
    the highest-F1 threshold overall. The test set is never used for this choice.
    """
    precision, recall, thr = precision_recall_curve(y_val, p_val)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision),
                   where=(precision + recall) > 0)
    ok = recall >= min_recall
    if ok.any():
        best = np.flatnonzero(ok)[np.argmax(f1[ok])]
    else:
        best = int(np.argmax(f1))
    return float(thr[best])


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and R^2."""
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
    }
