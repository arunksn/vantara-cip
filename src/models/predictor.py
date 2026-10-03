"""Single scoring implementation used by the training pipeline, batch scoring and the live API."""
from __future__ import annotations

import json
import os
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.explainability.explain import explain_in_words, lime_explain, make_explainer, shap_matrix  # noqa: E402
from src.utils.config import artifacts_path  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


class Predictor:
    """Loads persisted artifacts and scores customer feature records."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.meta: dict[str, Any] = json.loads(artifacts_path(cfg, "metadata.json").read_text(encoding="utf-8"))
        self.transformer = joblib.load(artifacts_path(cfg, "transformer.joblib"))
        self.features: list[str] = self.meta["model_features"]
        self.threshold: float = self.meta["churn_threshold"]
        self.churn_model = joblib.load(artifacts_path(cfg, "production_model.joblib"))
        self.clv_model = joblib.load(artifacts_path(cfg, "classical", "clv_model.joblib"))
        self.cat_model = joblib.load(artifacts_path(cfg, "classical", "next_category_model.joblib"))
        self.background: pd.DataFrame = joblib.load(artifacts_path(cfg, "explain_background.joblib"))
        self.medians: dict[str, float] = self.meta["feature_medians"]
        anom = joblib.load(artifacts_path(cfg, "deep_learning", "anomaly.joblib"))
        self._anom_pre, self._anom_thr, self._anom_cols = anom["preprocessor"], anom["threshold"], anom["features"]
        self._anom_scale = anom["resid_scale"]
        from tensorflow import keras  # lazy: heavy import

        self._ae = keras.models.load_model(artifacts_path(cfg, "deep_learning", "autoencoder.keras"), compile=False)
        self._clf = self.churn_model.named_steps["clf"] if hasattr(self.churn_model, "named_steps") else self.churn_model
        self._explainer = make_explainer(self._clf)

    # ------------------------------------------------------------------ scoring
    def matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw customer-feature records -> model matrix."""
        return self.transformer.transform(df)[self.features]

    def risk_tier(self, p: np.ndarray) -> np.ndarray:
        """Low / Medium / High from the validated operating threshold."""
        med = self.cfg["serving"]["risk_medium_ratio"] * self.threshold
        return np.where(p >= self.threshold, "High", np.where(p >= med, "Medium", "Low"))

    def anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Autoencoder reconstruction error on spending-pattern features."""
        z = self._anom_pre.transform(df[self._anom_cols].clip(lower=0)).astype("float32")
        recon = np.asarray(self._ae(z, training=False))
        return np.mean(((recon - z) ** 2) / self._anom_scale, axis=1)

    def predict(self, df: pd.DataFrame, with_shap: bool = False) -> pd.DataFrame:
        """Score a frame of customer feature records."""
        X = self.matrix(df)
        p = self.churn_model.predict_proba(X)[:, 1]
        clv = np.clip(self.clv_model.predict(X), 0, None)
        err = self.anomaly_scores(df)
        cat_p = self.cat_model.predict_proba(X)
        classes = list(self.cat_model.classes_)
        top3 = [[classes[j] for j in np.argsort(row)[::-1][:3]] for row in cat_p]
        out = pd.DataFrame({
            "churn_probability": p, "churn_predicted": (p >= self.threshold).astype(int),
            "risk_tier": self.risk_tier(p), "predicted_clv_90d": clv,
            "next_purchase_probability": 1.0 - p, "anomaly_score": err,
            "is_anomaly": err > self._anom_thr, "recommended_categories": top3,
        }, index=df.index)
        if with_shap:
            sv, _ = shap_matrix(self._explainer, X)
            k = self.cfg["explainability"]["top_reasons"]
            out["shap_json"] = [json.dumps({f: round(float(v), 5) for f, v in zip(self.features, row)}) for row in sv]
            out["top_reasons"] = [
                explain_in_words(dict(zip(self.features, row)), X.iloc[i].to_dict(), self.medians, p[i], self.threshold, k)
                for i, row in enumerate(sv)
            ]
        return out

    # ------------------------------------------------------------------ explanation
    def explain(self, record: pd.DataFrame, method: str = "shap") -> dict[str, Any]:
        """Per-feature contributions + plain-language text for ONE customer record (1-row frame)."""
        X = self.matrix(record)
        p = float(self.churn_model.predict_proba(X)[0, 1])
        if method == "lime":
            pairs = lime_explain(self._clf, self.background, X.iloc[0], self.cfg)
            contribs = dict(pairs)
        else:
            sv, _ = shap_matrix(self._explainer, X)
            contribs = dict(zip(self.features, map(float, sv[0])))
        text = explain_in_words(contribs, X.iloc[0].to_dict(), self.medians, p, self.threshold,
                                self.cfg["explainability"]["top_reasons"])
        rows = sorted(({"feature": f, "value": float(X.iloc[0][f]) if f in X.columns else None, "contribution": c}
                       for f, c in contribs.items()), key=lambda r: abs(r["contribution"]), reverse=True)
        return {"method": method, "churn_probability": p, "contributions": rows, "explanation": text}
