"""Explainability: SHAP (global + local), LIME comparison, partial dependence, plain-language explanations."""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402
from sklearn.inspection import PartialDependenceDisplay  # noqa: E402

from src.utils.config import path_of  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

FRIENDLY = {
    "recency_days": "days since last purchase", "frequency": "number of orders", "total_spend": "total spend",
    "avg_spend": "average order value", "historical_clv": "net lifetime revenue", "avg_basket_size": "average basket size",
    "freq_trend": "ordering trend", "gap_variance": "irregularity of time between orders",
    "seasonal_concentration": "seasonality of purchases", "return_rate": "return rate",
    "discount_sensitivity": "reliance on discounted items", "avg_product_popularity": "preference for popular products",
    "engagement_score": "overall engagement score",
}


def friendly(name: str) -> str:
    """Human-readable feature name."""
    if name in FRIENDLY:
        return FRIENDLY[name]
    if name.startswith("affinity_"):
        return "share of spend on " + name[9:].replace("_", " ")
    if name.startswith("country_"):
        return "country (" + name[8:] + ")"
    return name.replace("_", " ")


def make_explainer(clf: Any) -> shap.TreeExplainer:
    """TreeExplainer for the production (tree-based) churn model."""
    return shap.TreeExplainer(clf)


def shap_matrix(explainer: shap.TreeExplainer, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP values for the churn (positive) class and the base value."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    base = np.ravel(explainer.expected_value)
    return sv, float(base[-1])


def explain_in_words(
    contribs: dict[str, float], values: dict[str, float], medians: dict[str, float], proba: float,
    threshold: float, n: int = 3,
) -> str:
    """Plain-language 'why this customer is flagged' sentence (no ML background required)."""
    if proba >= threshold:
        head = (f"This customer is flagged as HIGH churn risk: predicted churn probability {proba:.0%}, at or above the "
                f"{threshold:.0%} operating threshold")
    elif proba >= 0.6 * threshold:
        head = f"This customer has MEDIUM churn risk: predicted churn probability {proba:.0%} (threshold {threshold:.0%})"
    else:
        head = f"This customer is LOW churn risk: predicted churn probability {proba:.0%} (threshold {threshold:.0%})"
    top = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
    parts = []
    for feat, c in top:
        v, med = values.get(feat), medians.get(feat)
        rel = ""
        if v is not None and med is not None and not feat.startswith("country_"):
            rel = " (higher than typical)" if v > med else " (lower than typical)" if v < med else " (typical)"
        direction = "raises" if c > 0 else "lowers"
        val = "" if v is None else f" = {v:,.2f}"
        parts.append(f"{friendly(feat)}{val}{rel} {direction} the risk")
    return head + ". Main drivers (relative to the average customer): " + "; ".join(parts) + "."


def global_importance(sv: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature, sorted."""
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)})
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def save_summary_plot(sv: np.ndarray, X: pd.DataFrame, path: Any) -> None:
    """Global SHAP summary (beeswarm) plot."""
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")


def _bar_fallback(contribs: np.ndarray, names: list[str], title: str, path: Any) -> None:
    order = np.argsort(np.abs(contribs))[::-1][:10][::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh([names[i] for i in order], contribs[order], color=["#d62728" if contribs[i] > 0 else "#1f77b4" for i in order])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_force_plot(base: float, row_sv: np.ndarray, row: pd.Series, title: str, path: Any) -> None:
    """SHAP force plot for one customer (matplotlib); falls back to a contribution bar chart."""
    try:
        shap.force_plot(base, row_sv, row.round(2), matplotlib=True, show=False, text_rotation=15)
        plt.title(title, fontsize=9)
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:  # noqa: BLE001
        logger.warning("force plot failed (%s); writing bar chart instead", exc)
        plt.close("all")
        _bar_fallback(row_sv, list(row.index), title, path)


def pick_representative(proba: np.ndarray, threshold: float) -> dict[str, int]:
    """Positional indices of one low-risk, one high-risk and one borderline customer."""
    return {"low_risk": int(np.argmin(proba)), "high_risk": int(np.argmax(proba)),
            "borderline": int(np.argmin(np.abs(proba - threshold)))}


def lime_explain(clf: Any, background: pd.DataFrame, row: pd.Series, cfg: dict[str, Any]) -> list[tuple[str, float]]:
    """Local LIME explanation for one customer as (feature, weight) pairs."""
    ecfg = cfg["explainability"]
    names = list(background.columns)
    expl = LimeTabularExplainer(background.to_numpy(float), feature_names=names, class_names=["retained", "churn"],
                                mode="classification", discretize_continuous=True, random_state=cfg["project"]["seed"])

    def predict_fn(a: np.ndarray) -> np.ndarray:
        return clf.predict_proba(pd.DataFrame(a, columns=names))

    exp = expl.explain_instance(row.to_numpy(float), predict_fn, num_features=ecfg["lime_num_features"],
                                num_samples=ecfg["lime_num_samples"], labels=(1,))
    return [(names[i], float(w)) for i, w in exp.as_map()[1]]


def compare_lime_shap(lime_pairs: list[tuple[str, float]], shap_row: np.ndarray, names: list[str], k: int = 5) -> dict[str, Any]:
    """Agreement between LIME and SHAP: top-k overlap and sign agreement on shared features."""
    shap_top = [names[i] for i in np.argsort(np.abs(shap_row))[::-1][:k]]
    lime_top = [f for f, _ in sorted(lime_pairs, key=lambda kv: abs(kv[1]), reverse=True)[:k]]
    overlap = sorted(set(shap_top) & set(lime_top))
    lime_d = dict(lime_pairs)
    shared = [f for f in lime_d if f in names]
    sign_agree = [np.sign(lime_d[f]) == np.sign(shap_row[names.index(f)]) for f in shared if shap_row[names.index(f)] != 0]
    return {"top_k": k, "shap_top": shap_top, "lime_top": lime_top, "overlap": overlap,
            "overlap_fraction": len(overlap) / k, "sign_agreement": float(np.mean(sign_agree)) if sign_agree else None}


def save_lime_vs_shap_plot(lime_pairs: list[tuple[str, float]], shap_row: np.ndarray, names: list[str], path: Any) -> None:
    """Side-by-side bars of LIME weights and SHAP values for the same customer."""
    feats = [f for f, _ in sorted(lime_pairs, key=lambda kv: abs(kv[1]), reverse=True)[:8]][::-1]
    lw = dict(lime_pairs)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    axs[0].barh(feats, [lw[f] for f in feats], color="#9467bd")
    axs[0].set_title("LIME weight")
    axs[1].barh(feats, [shap_row[names.index(f)] for f in feats], color="#2ca02c")
    axs[1].set_title("SHAP value")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_pdp(clf: Any, X: pd.DataFrame, features: list[str], path: Any) -> None:
    """Partial dependence plots for the most influential features."""
    fig, ax = plt.subplots(1, len(features), figsize=(4.2 * len(features), 3.6))
    ax = np.atleast_1d(ax)
    PartialDependenceDisplay.from_estimator(clf, X, features, ax=list(ax), kind="average")
    fig.suptitle("Partial dependence on predicted churn probability", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figures_dir(cfg: dict[str, Any]) -> Any:
    """Figures output directory."""
    p = path_of(cfg, "figures_dir")
    p.mkdir(parents=True, exist_ok=True)
    return p
