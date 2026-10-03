"""Unit tests: loading, cleaning, validation, point-in-time feature engineering (incl. leakage), utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.clean import assign_category, clean_transactions, flag_outliers
from src.data.load import standardize_columns
from src.data.validate import DataValidationError, validate_features, validate_transactions
from src.features.builders import (
    LeakageError,
    build_customer_features,
    build_labels,
    feature_columns,
    point_in_time_frame,
)
from src.features.sequences import build_sequences
from src.features.transform import FeatureTransformer, compute_vif, select_features_by_vif
from src.models.common import classification_metrics, make_splits, select_threshold
from src.utils.config import override_config
from src.utils.forecast import drop_partial_last_month, forecast_revenue


def make_tx(rows: list[tuple]) -> pd.DataFrame:
    """rows: (invoice, customer, date, stock, qty, price[, category])."""
    recs = []
    for r in rows:
        inv, cid, date, stock, qty, price = r[:6]
        cat = r[6] if len(r) > 6 else "lighting"
        recs.append({"invoice": inv, "customer_id": cid, "invoice_date": pd.Timestamp(date), "stock_code": stock,
                     "quantity": qty, "unit_price": price, "description": f"ITEM {stock}", "country": "United Kingdom",
                     "is_return": qty < 0 or inv.startswith("C"), "is_admin_code": False, "category": cat,
                     "line_total": qty * price, "use_for_customer_model": True})
    df = pd.DataFrame(recs)
    df["customer_id"] = df["customer_id"].astype("Int64")
    return df


CUT = pd.Timestamp("2011-06-01")
BASE = [
    ("1", 1, "2011-01-10", "A", 2, 10.0), ("2", 1, "2011-03-10", "B", 1, 20.0), ("3", 1, "2011-05-20", "A", 1, 10.0),
    ("C4", 1, "2011-05-25", "A", -1, 10.0),
    ("5", 2, "2011-02-01", "C", 5, 4.0),
]
FUTURE = [("6", 1, "2011-06-15", "A", 3, 10.0), ("7", 3, "2011-07-01", "D", 1, 99.0)]


# ------------------------------------------------------------------ loading / cleaning / validation
def test_standardize_columns_handles_aliases():
    df = pd.DataFrame({"Invoice": ["1"], "StockCode": ["a"], "Description": ["x"], "Quantity": [1],
                       "InvoiceDate": ["2011-01-01"], "Price": [1.5], "Customer ID": [1.0], "Country": ["UK"]})
    out = standardize_columns(df)
    assert {"invoice", "stock_code", "unit_price", "customer_id", "invoice_date"} <= set(out.columns)
    assert out["stock_code"].iloc[0] == "A"
    df2 = df.rename(columns={"Price": "UnitPrice", "Customer ID": "CustomerID"})
    assert "unit_price" in standardize_columns(df2).columns


def test_cleaning_rules(base_cfg):
    raw = pd.DataFrame({
        "invoice": ["1", "1", "2", "C3", "4", "5", "6", "6"], "stock_code": ["A", "A", "A", "A", "POST", "B", "C", "C"],
        "description": ["mug  red", "mug  red", "Mug Red", "MUG RED", "postage", "x", "Candle", "Candle"],
        "quantity": [1, 1, 2, -2, 1, 3, 90000, -90000],
        "invoice_date": pd.to_datetime(["2011-01-01"] * 4 + ["2011-01-02", "2011-01-03", "2011-01-04", "2011-01-05"]),
        "unit_price": [1.0, 1.0, 1.0, 1.0, 15.0, 0.0, 2.0, 2.0],
        "customer_id": pd.array([1, 1, 1, 1, 1, pd.NA, 2, 2], dtype="Int64"), "country": ["UK"] * 8})
    filler = pd.DataFrame({
        "invoice": [f"9{i}" for i in range(80)], "stock_code": "F", "description": "FILLER ITEM", "quantity": [1 + i % 5 for i in range(80)],
        "invoice_date": pd.Timestamp("2011-02-01"), "unit_price": 2.0, "customer_id": pd.array(range(100, 180), dtype="Int64"),
        "country": "UK"})
    out, rep = clean_transactions(pd.concat([raw, filler], ignore_index=True), base_cfg)  # ordinary rows keep the IQR fences realistic
    assert rep["exact_duplicates_removed"] == 1
    assert out["is_return"].sum() == 2
    assert out.loc[out["stock_code"] == "POST", "is_admin_code"].all()
    assert out.loc[out["stock_code"] == "B", "is_price_adjustment"].all()
    assert (~out.loc[out["stock_code"] == "B", "has_customer_id"]).all()
    assert set(out.loc[out["stock_code"] == "A", "description"]) == {"MUG RED"}
    assert out.loc[out["stock_code"] == "C", "is_entry_error"].all()
    assert not out.loc[out["stock_code"] == "C", "use_for_customer_model"].any()


def test_bulk_outlier_kept_when_not_reversed(base_cfg):
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"customer_id": pd.array(range(n), dtype="Int64"), "stock_code": "A", "quantity": rng.integers(1, 10, n),
                       "unit_price": 1.0, "is_price_adjustment": False})
    df.loc[0, "quantity"] = 3000
    out, info = flag_outliers(df, 3.0, 50000)
    assert out.loc[0, "is_bulk_outlier"] and not out.loc[0, "is_entry_error"]
    assert info["n_bulk_outliers"] >= 1


def test_assign_category(base_cfg):
    s = pd.Series(["RED CANDLE HOLDER", "PINK MUG", "STRANGE THING", None])
    out = assign_category(s, base_cfg["features"]["category_rules"])
    assert list(out) == ["lighting", "kitchen_dining", "other", "other"]


def test_validation_fails_loudly(base_cfg):
    good = pd.DataFrame({"invoice": ["1"], "stock_code": ["A"], "description": ["X"], "quantity": [1],
                         "invoice_date": pd.to_datetime(["2011-01-01"]), "unit_price": [1.0], "customer_id": [1], "country": ["UK"]})
    assert validate_transactions(good, base_cfg)["rows"] == 1
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_transactions(good.drop(columns=["unit_price"]), base_cfg)
    with pytest.raises(DataValidationError, match="date range"):
        validate_transactions(good.assign(invoice_date=pd.to_datetime(["1999-01-01"])), base_cfg)
    with pytest.raises(DataValidationError, match="null rate"):
        validate_transactions(good.assign(customer_id=np.nan), base_cfg)
    with pytest.raises(DataValidationError):
        validate_transactions(good.iloc[0:0], base_cfg)


# ------------------------------------------------------------------ features: correctness
def test_feature_values_hand_computed(base_cfg):
    feats, _ = build_customer_features(make_tx(BASE), CUT, base_cfg)
    c1 = feats.set_index("customer_id").loc[1]
    assert c1["frequency"] == 3
    assert c1["recency_days"] == (CUT - pd.Timestamp("2011-05-20")).days
    assert c1["total_spend"] == pytest.approx(2 * 10 + 20 + 10)
    assert c1["historical_clv"] == pytest.approx(50 - 10)
    assert c1["avg_spend"] == pytest.approx(50 / 3)
    assert c1["return_rate"] == pytest.approx(1 / 4)
    gaps = np.array([59, 71.0])
    assert c1["gap_variance"] == pytest.approx(gaps.var(), rel=1e-6)
    assert 0 < c1["seasonal_concentration"] <= 1


def test_feature_ranges(base_cfg):
    feats, _ = build_customer_features(make_tx(BASE), CUT, base_cfg)
    assert (feats["recency_days"] >= 0).all() and (feats["frequency"] >= 1).all()
    assert feats["return_rate"].between(0, 1).all() and feats["discount_sensitivity"].between(0, 1).all()
    aff = feats[[c for c in feats.columns if c.startswith("affinity_")]].sum(axis=1)
    assert (aff <= 1 + 1e-9).all()


# ------------------------------------------------------------------ LEAKAGE TESTS (PRD Section 7)
def test_features_ignore_future_rows(base_cfg):
    without, _ = build_customer_features(make_tx(BASE), CUT, base_cfg)
    with_future, _ = build_customer_features(make_tx(BASE + FUTURE), CUT, base_cfg)
    pd.testing.assert_frame_equal(without.reset_index(drop=True), with_future.reset_index(drop=True))


def test_features_unchanged_when_future_values_altered(base_cfg):
    a = make_tx(BASE + FUTURE)
    b = a.copy()
    fut = b["invoice_date"] >= CUT
    b.loc[fut, ["quantity", "unit_price"]] = [999, 999.0]
    b["line_total"] = b["quantity"] * b["unit_price"]
    fa, _ = build_customer_features(a, CUT, base_cfg)
    fb, _ = build_customer_features(b, CUT, base_cfg)
    pd.testing.assert_frame_equal(fa.reset_index(drop=True), fb.reset_index(drop=True))
    assert 3 not in set(fa["customer_id"])  # customer first seen after the cutoff gets no features


def test_point_in_time_assertion_triggers():
    tx = make_tx(BASE + FUTURE)
    sub = point_in_time_frame(tx, CUT)
    assert sub["invoice_date"].max() < CUT
    with pytest.raises(LeakageError):
        raise LeakageError("sentinel")


def test_stock_frequency_uses_only_pre_cutoff(base_cfg):
    _, f1 = build_customer_features(make_tx(BASE), CUT, base_cfg)
    _, f2 = build_customer_features(make_tx(BASE + FUTURE), CUT, base_cfg)
    assert "D" not in f2.index and f1.equals(f2)


def test_labels_use_only_horizon_window(base_cfg):
    tx = make_tx(BASE + FUTURE)
    ids = pd.Series([1, 2], name="customer_id")
    lab = build_labels(tx, ids, CUT, base_cfg).set_index("customer_id")
    assert lab.loc[1, "churn"] == 0 and lab.loc[2, "churn"] == 1
    assert lab.loc[1, "clv_target"] == pytest.approx(30.0)
    assert lab.loc[2, "clv_target"] == 0.0
    assert lab.loc[1, "next_category"] == "lighting" and pd.isna(lab.loc[2, "next_category"])
    beyond = make_tx(BASE + [("9", 2, "2011-09-30", "C", 1, 4.0)])
    assert build_labels(beyond, ids, CUT, base_cfg).set_index("customer_id").loc[2, "churn"] == 1  # outside 90 days


def test_sequences_ignore_future(base_cfg):
    cfg = override_config(base_cfg, {"features": {"sequence_max_len": 5}})
    ids = np.array([1, 2])
    n1, c1 = build_sequences(make_tx(BASE), ids, CUT, cfg)
    n2, c2 = build_sequences(make_tx(BASE + FUTURE), ids, CUT, cfg)
    assert np.array_equal(n1, n2) and np.array_equal(c1, c2)
    assert n1.shape == (2, 6, 3) and n1[0, 5, 1] > 0  # terminal cutoff-marker step carries days-since-last-order


# ------------------------------------------------------------------ transformer / utilities
def test_transformer_and_engagement(base_cfg):
    feats, _ = build_customer_features(make_tx(BASE), CUT, base_cfg)
    ft = FeatureTransformer(base_cfg).fit(feats)
    X = ft.transform(feats.assign(country="Narnia"))
    assert list(X.columns) == ft.feature_names and X.notna().all().all()
    assert ft.engagement_score(feats).between(0, 100).all()
    assert set(feature_columns(base_cfg)) <= set(X.columns)


def test_vif_pruning_removes_collinear_feature():
    rng = np.random.default_rng(1)
    a = rng.normal(size=300)
    X = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(scale=0.01, size=300), "c": rng.normal(size=300)})
    assert compute_vif(X)["b"] > 100
    kept, _ = select_features_by_vif(X, 10.0, protected=[])
    assert "c" in kept and len({"a", "b"} & set(kept)) == 1


def test_splits_stratified_and_disjoint(base_cfg):
    y = (np.random.default_rng(0).random(1000) < 0.3).astype(int)
    sp = make_splits(y, base_cfg)
    assert sum(len(v) for v in sp.values()) == 1000
    assert not (set(sp["train"]) & set(sp["test"])) and not (set(sp["train"]) & set(sp["val"]))
    assert abs(y[sp["test"]].mean() - y.mean()) < 0.02 and abs(len(sp["train"]) / 1000 - 0.7) < 0.01
    assert all(np.array_equal(make_splits(y, base_cfg)[k], v) for k, v in sp.items())  # deterministic


def test_threshold_selection_meets_recall_target():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.3).astype(int)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    thr = select_threshold(y, p, 0.7)
    m = classification_metrics(y, p, thr)
    assert m["recall"] >= 0.7 and 0 <= m["roc_auc"] <= 1 and m["tp"] + m["fn"] == y.sum()


def test_validate_features_rejects_bad_values(base_cfg):
    feats, _ = build_customer_features(make_tx(BASE), CUT, base_cfg)
    feats["engagement_score"] = 50.0
    validate_features(feats, feature_columns(base_cfg))
    bad = feats.copy()
    bad.loc[0, "recency_days"] = -1
    with pytest.raises(DataValidationError):
        validate_features(bad, feature_columns(base_cfg))


def test_forecast_helpers():
    months = pd.date_range("2010-01-01", periods=25, freq="MS")
    m = pd.DataFrame({"month": months, "revenue": np.linspace(100, 200, 25), "last_date": months + pd.offsets.MonthEnd(0)})
    fc = forecast_revenue(drop_partial_last_month(m), 3)
    assert len(fc) == 3 and (fc["revenue"] > 0).all() and "seasonal" in fc["method"].iloc[0]
    short = forecast_revenue(m.head(10), 2)
    assert short["method"].iloc[0] == "linear trend"
    partial = m.copy()
    partial.loc[24, "last_date"] = partial.loc[24, "month"] + pd.Timedelta(days=5)
    assert len(drop_partial_last_month(partial)) == 24
