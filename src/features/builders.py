"""Customer-level feature engineering with strict point-in-time cutoff discipline.

Every feature uses only transactions with invoice_date STRICTLY BEFORE the cutoff. Labels (churn, future
spend, next category) use only the horizon window starting AT the cutoff. See tests/test_features.py.

Feature -> justification (one line each):
  recency_days             days since last purchase; strongest univariate churn predictor
  frequency                distinct orders; separates habitual buyers from one-time purchasers
  total_spend / avg_spend  gross monetary value; core input for churn weighting and CLV
  historical_clv           net revenue (sales minus returns) to date; value-based prioritisation
  avg_basket_size          units per order; top-up vs bulk shopper
  freq_trend               slope of orders per 30-day bin; captures acceleration/deceleration
  gap_variance             variance of days between orders; irregular buyers are harder to retain
  seasonal_concentration   Herfindahl index of calendar-month order shares; gift/seasonal vs year-round
  affinity_<category>      share of spend per product category; feeds recommendations
  return_rate              share of lines that are returns; leading indicator of dissatisfaction
  discount_sensitivity     share of spend on lines >=10% below SKU median price (proxy; the dataset has no
                           explicit discount field)
  avg_product_popularity   frequency-encoded StockCode (share of customers buying each SKU), averaged
  engagement_score         composite 0-100 of recency, frequency, monetary percentile ranks
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

BASE_FEATURES = [
    "recency_days", "frequency", "total_spend", "avg_spend", "historical_clv", "avg_basket_size",
    "freq_trend", "gap_variance", "seasonal_concentration", "return_rate", "discount_sensitivity",
    "avg_product_popularity", "engagement_score",
]
META_COLUMNS = ["customer_id", "country", "first_purchase", "last_purchase"]
LABEL_COLUMNS = ["churn", "clv_target", "next_category"]


class LeakageError(AssertionError):
    """Raised when a feature computation would use data on/after the cutoff."""


def affinity_columns(cfg: dict[str, Any]) -> list[str]:
    """Affinity feature names (one per category rule; 'other' is the implied remainder)."""
    return [f"affinity_{c}" for c in cfg["features"]["category_rules"]]


def feature_columns(cfg: dict[str, Any]) -> list[str]:
    """All numeric engineered feature columns (before country one-hot encoding)."""
    return BASE_FEATURES + affinity_columns(cfg)


def compute_cutoffs(tx: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (training cutoff, scoring cutoff).

    scoring cutoff = day after the last transaction; training cutoff = scoring cutoff - churn horizon.
    """
    scoring = tx["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    train = scoring - pd.Timedelta(days=cfg["features"]["churn_horizon_days"])
    return train, scoring


def point_in_time_frame(tx: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Restrict to customer-model rows strictly before cutoff, and assert it."""
    sub = tx[tx["use_for_customer_model"] & (tx["invoice_date"] < cutoff)]
    if len(sub) and sub["invoice_date"].max() >= cutoff:
        raise LeakageError("transactions on/after the cutoff reached feature computation")
    return sub


def fit_stock_frequency(sales: pd.DataFrame) -> pd.Series:
    """Frequency-encode StockCode: share of customers who bought each (non-admin) SKU."""
    prod = sales[~sales["is_admin_code"]]
    n_cust = max(prod["customer_id"].nunique(), 1)
    return prod.drop_duplicates(["customer_id", "stock_code"]).groupby("stock_code").size() / n_cust


def _trend_slope(orders: pd.DataFrame, cutoff: pd.Timestamp, n_bins: int, bin_days: int,
                 first_order: pd.Series, index: pd.Index) -> pd.Series:
    """Least-squares slope of orders-per-bin over the bins since the customer's first order."""
    age = ((cutoff - orders["order_date"]).dt.days // bin_days).clip(0, n_bins - 1).to_numpy()
    cid = index.get_indexer(orders["customer_id"])
    counts = np.zeros((len(index), n_bins))
    np.add.at(counts, (cid, age), 1)
    first_bin = ((cutoff - first_order.reindex(index)).dt.days // bin_days).clip(0, n_bins - 1).to_numpy()
    bins = np.arange(n_bins)
    mask = bins[None, :] <= first_bin[:, None]
    t = (n_bins - 1 - bins)[None, :] * np.ones((len(index), 1))
    n_valid = mask.sum(1)
    t_mean = (mask * t).sum(1) / n_valid
    c_mean = (mask * counts).sum(1) / n_valid
    cov = (mask * (t - t_mean[:, None]) * (counts - c_mean[:, None])).sum(1)
    var = (mask * (t - t_mean[:, None]) ** 2).sum(1)
    slope = np.divide(cov, var, out=np.zeros_like(cov), where=var > 0)
    return pd.Series(slope, index=index)


def build_customer_features(
    tx: pd.DataFrame,
    cutoff: pd.Timestamp,
    cfg: dict[str, Any],
    stock_freq: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute customer-level features using only data strictly before ``cutoff``.

    Returns (features indexed by customer_id with META columns, fitted StockCode frequency map).
    """
    fcfg = cfg["features"]
    sub = point_in_time_frame(tx, cutoff)
    sales = sub[~sub["is_return"]]
    rets = sub[sub["is_return"]]
    if stock_freq is None:
        stock_freq = fit_stock_frequency(sales)

    orders = (
        sales.groupby(["customer_id", "invoice"], sort=False)
        .agg(order_date=("invoice_date", "max"), order_value=("line_total", "sum"),
             units=("quantity", "sum"), country=("country", "last"))
        .reset_index()
        .sort_values(["customer_id", "order_date"], kind="stable")
    )
    g = orders.groupby("customer_id")
    feats = pd.DataFrame(index=g.size().index)
    feats.index.name = "customer_id"
    feats["first_purchase"] = g["order_date"].min()
    feats["last_purchase"] = g["order_date"].max()
    feats["recency_days"] = (cutoff - feats["last_purchase"]).dt.days.astype(float)
    feats["frequency"] = g.size().astype(float)
    feats["total_spend"] = g["order_value"].sum().clip(lower=0)
    feats["avg_spend"] = feats["total_spend"] / feats["frequency"]
    feats["avg_basket_size"] = g["units"].mean()
    feats["country"] = g["country"].last().astype(str)

    net = sub.groupby("customer_id")["line_total"].sum()
    feats["historical_clv"] = net.reindex(feats.index).fillna(0.0).clip(lower=0)

    feats["freq_trend"] = _trend_slope(orders, cutoff, fcfg["trend_window_bins"], fcfg["trend_bin_days"],
                                       feats["first_purchase"], feats.index)

    orders["gap_days"] = orders.groupby("customer_id")["order_date"].diff().dt.total_seconds() / 86400.0
    feats["gap_variance"] = orders.groupby("customer_id")["gap_days"].var(ddof=0).reindex(feats.index).fillna(0.0)

    orders["month"] = orders["order_date"].dt.month
    mshare = orders.groupby(["customer_id", "month"]).size().unstack(fill_value=0)
    mshare = mshare.div(mshare.sum(1), axis=0)
    feats["seasonal_concentration"] = (mshare**2).sum(1).reindex(feats.index)

    n_ret = rets.groupby("customer_id").size().reindex(feats.index).fillna(0)
    n_lines = sub.groupby("customer_id").size().reindex(feats.index).fillna(1)
    feats["return_rate"] = (n_ret / n_lines).clip(0, 1)

    prod_sales = sales[~sales["is_admin_code"]].copy()
    med = prod_sales.groupby("stock_code")["unit_price"].transform("median")
    prod_sales["discounted"] = prod_sales["unit_price"] < (1 - fcfg["discount_threshold"]) * med
    prod_sales["disc_spend"] = np.where(prod_sales["discounted"], prod_sales["line_total"], 0.0)
    tot = prod_sales.groupby("customer_id")["line_total"].sum()
    disc = prod_sales.groupby("customer_id")["disc_spend"].sum()
    feats["discount_sensitivity"] = (disc / tot.where(tot > 0)).reindex(feats.index).fillna(0.0).clip(0, 1)

    pop = prod_sales.drop_duplicates(["customer_id", "stock_code"]).copy()
    pop["pop"] = pop["stock_code"].map(stock_freq).fillna(0.0)
    feats["avg_product_popularity"] = pop.groupby("customer_id")["pop"].mean().reindex(feats.index).fillna(0.0)

    cat_spend = prod_sales.groupby(["customer_id", "category"])["line_total"].sum().unstack(fill_value=0.0)
    cat_total = cat_spend.sum(1).where(lambda s: s > 0)
    cat_share = cat_spend.div(cat_total, axis=0).fillna(0.0)
    for cat in fcfg["category_rules"]:
        col = cat_share[cat] if cat in cat_share.columns else 0.0
        feats[f"affinity_{cat}"] = pd.Series(col, index=cat_share.index).reindex(feats.index).fillna(0.0)

    feats["engagement_score"] = np.nan  # filled from the fitted FeatureTransformer reference
    feats = feats.reset_index()
    logger.info("built %d customer feature rows at cutoff %s", len(feats), cutoff)
    return feats, stock_freq


def build_labels(
    tx: pd.DataFrame, customer_ids: pd.Series, cutoff: pd.Timestamp, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Churn / future-spend / next-category labels from the horizon window [cutoff, cutoff + horizon)."""
    fcfg = cfg["features"]
    end = cutoff + pd.Timedelta(days=fcfg["churn_horizon_days"])
    win = tx[tx["use_for_customer_model"] & (tx["invoice_date"] >= cutoff) & (tx["invoice_date"] < end)]
    sales = win[~win["is_return"]]
    spend = sales.groupby("customer_id")["line_total"].sum()
    out = pd.DataFrame({"customer_id": customer_ids.to_numpy()}).set_index("customer_id")
    out["churn"] = (~out.index.isin(sales["customer_id"].unique())).astype(int)
    clv_end = cutoff + pd.Timedelta(days=fcfg["clv_horizon_days"])
    clv_sales = sales[sales["invoice_date"] < clv_end]
    out["clv_target"] = clv_sales.groupby("customer_id")["line_total"].sum().reindex(out.index).fillna(0.0)
    prod = sales[~sales["is_admin_code"]]
    cat_spend = prod.groupby(["customer_id", "category"])["line_total"].sum().reset_index()
    top = cat_spend.sort_values(["customer_id", "line_total"], ascending=[True, False]).drop_duplicates("customer_id")
    out["next_category"] = top.set_index("customer_id")["category"].reindex(out.index)
    del spend
    return out.reset_index()


def make_train_and_scoring_tables(
    tx: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, str]]:
    """Build the labelled training snapshot and the unlabelled scoring snapshot."""
    train_cut, score_cut = compute_cutoffs(tx, cfg)
    train_feats, stock_freq = build_customer_features(tx, train_cut, cfg)
    labels = build_labels(tx, train_feats["customer_id"], train_cut, cfg)
    train_tbl = train_feats.merge(labels, on="customer_id", how="left")
    score_feats, _ = build_customer_features(tx, score_cut, cfg, stock_freq=stock_freq)
    return train_tbl, score_feats, stock_freq, {"train_cutoff": str(train_cut), "scoring_cutoff": str(score_cut)}
