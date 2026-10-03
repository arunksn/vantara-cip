"""Fitted feature transformer shared by the training pipeline, batch scoring and the live API.

Persisting the fitted object guarantees the exact same encoding/scaling at inference time.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.features.builders import affinity_columns, feature_columns


class FeatureTransformer:
    """Country one-hot encoding + engagement score + fixed column ordering."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.top_n = int(cfg["features"]["top_countries"])
        self.numeric_cols = feature_columns(cfg)
        self.affinity_cols = affinity_columns(cfg)
        self.country_levels: list[str] = []
        self.engagement_ref: dict[str, np.ndarray] = {}
        self.fitted = False

    # ---- engagement ---------------------------------------------------------------
    def _pct(self, key: str, values: np.ndarray) -> np.ndarray:
        ref = self.engagement_ref[key]
        return np.searchsorted(ref, values, side="right") / len(ref)

    def engagement_score(self, df: pd.DataFrame) -> pd.Series:
        """Composite 0-100 score from percentile ranks of recency (inverted), frequency, monetary."""
        r = 1.0 - self._pct("recency_days", df["recency_days"].to_numpy(float))
        f = self._pct("frequency", df["frequency"].to_numpy(float))
        m = self._pct("total_spend", df["total_spend"].to_numpy(float))
        return pd.Series(100.0 * (r + f + m) / 3.0, index=df.index)

    # ---- fit / transform ------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> FeatureTransformer:
        """Learn country levels and engagement reference distributions (features only, no labels)."""
        counts = df["country"].astype(str).value_counts()
        self.country_levels = list(counts.index[: self.top_n]) + ["Other"]
        self.engagement_ref = {
            "recency_days": np.sort(df["recency_days"].to_numpy(float)),
            "frequency": np.sort(df["frequency"].to_numpy(float)),
            "total_spend": np.sort(df["total_spend"].to_numpy(float)),
        }
        self.fitted = True
        return self

    @property
    def country_columns(self) -> list[str]:
        """One-hot columns (the most frequent country is the dropped reference level)."""
        return [f"country_{c}" for c in self.country_levels[1:]]

    @property
    def feature_names(self) -> list[str]:
        """Ordered model-matrix column names."""
        return self.numeric_cols + self.country_columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the model feature matrix (float columns in fixed order)."""
        if not self.fitted:
            raise RuntimeError("FeatureTransformer must be fitted first")
        out = pd.DataFrame(index=df.index)
        work = df.copy()
        for col in self.affinity_cols:
            if col not in work.columns:
                work[col] = 0.0
        if "engagement_score" not in work.columns:
            work["engagement_score"] = np.nan
        missing = work["engagement_score"].isna()
        if missing.any():
            work.loc[missing, "engagement_score"] = self.engagement_score(work)[missing]
        for col in self.numeric_cols:
            out[col] = pd.to_numeric(work[col], errors="coerce").astype(float)
        grouped = work["country"].astype(str).where(work["country"].astype(str).isin(self.country_levels), "Other")
        for lvl in self.country_levels[1:]:
            out[f"country_{lvl}"] = (grouped == lvl).astype(float)
        return out[self.feature_names]


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factor for each column (inf for perfectly collinear columns)."""
    arr = np.column_stack([np.ones(len(X)), X.to_numpy(float)])
    vals = []
    for i in range(1, arr.shape[1]):
        try:
            vals.append(variance_inflation_factor(arr, i))
        except Exception:  # noqa: BLE001 - singular design
            vals.append(np.inf)
    return pd.Series(vals, index=X.columns)


def select_features_by_vif(
    X: pd.DataFrame, threshold: float, protected: list[str], skip_prefixes: tuple[str, ...] = ("country_",)
) -> tuple[list[str], pd.DataFrame]:
    """Iteratively drop the highest-VIF unprotected feature until all VIFs are below the threshold.

    Returns (kept columns, history table of dropped features with VIF at the time of dropping).
    """
    cols = [c for c in X.columns if not c.startswith(skip_prefixes)]
    skipped = [c for c in X.columns if c.startswith(skip_prefixes)]
    history = []
    while True:
        vif = compute_vif(X[cols].loc[:, X[cols].std() > 0]).replace(np.inf, 1e12)
        droppable = vif[~vif.index.isin(protected)]
        if droppable.empty or droppable.max() <= threshold:
            final = vif
            break
        worst = droppable.idxmax()
        history.append({"dropped": worst, "vif": float(droppable.max())})
        cols.remove(worst)
    table = final.rename("vif").reset_index().rename(columns={"index": "feature"})
    table["status"] = "kept"
    hist = pd.DataFrame(history, columns=["dropped", "vif"])
    kept = [c for c in X.columns if c in cols or c in skipped]
    return kept, pd.concat([table, hist.rename(columns={"dropped": "feature"}).assign(status="dropped")],
                           ignore_index=True)
