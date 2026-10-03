"""API integration tests against artifacts trained on a small synthetic dataset (see conftest.trained_env)."""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

GOOD = {"customer_id": 12346, "country": "United Kingdom", "recency_days": 45, "frequency": 6, "total_spend": 1450.5,
        "avg_spend": 241.75, "historical_clv": 1380.0, "avg_basket_size": 120, "freq_trend": -0.1, "gap_variance": 900,
        "seasonal_concentration": 0.2, "return_rate": 0.02, "discount_sensitivity": 0.1, "avg_product_popularity": 0.05,
        "category_affinity": {"lighting": 0.3, "kitchen_dining": 0.2}}


@pytest.fixture(scope="module")
def client(trained_env):
    from api.main import create_app

    cfg, tmp = trained_env
    with TestClient(create_app(database_url=f"sqlite:///{tmp}/api.db")) as c:
        yield c


def test_health_and_metadata(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] and h["database"] == "ok"
    m = client.get("/model/metadata").json()
    assert m["production_model"] in {"random_forest", "xgboost", "lightgbm"} and 0 < m["churn_threshold"] < 1
    assert m["model_features"] and m["data_source"]


def test_predict_customer(client):
    r = client.post("/predict/customer", json=GOOD)
    assert r.status_code == 200, r.text
    b = r.json()
    assert 0 <= b["churn_probability"] <= 1 and b["risk_tier"] in {"Low", "Medium", "High"}
    assert b["next_purchase_probability"] == pytest.approx(1 - b["churn_probability"])
    assert b["predicted_clv_90d"] >= 0 and len(b["recommended_categories"]) == 3


def test_recent_frequent_buyer_less_risky_than_lapsed(client):
    active = client.post("/predict/customer", json={**GOOD, "recency_days": 3, "frequency": 25}).json()
    lapsed = client.post("/predict/customer", json={**GOOD, "recency_days": 400, "frequency": 2}).json()
    assert lapsed["churn_probability"] > active["churn_probability"]


@pytest.mark.parametrize("patch,field", [
    ({"recency_days": -5}, "recency_days"), ({"return_rate": 1.5}, "return_rate"), ({"frequency": 0}, "frequency"),
    ({"country": ""}, "country"), ({"seasonal_concentration": 0}, "seasonal_concentration"),
    ({"category_affinity": {"lighting": 1.4}}, "category_affinity"), ({"bogus_field": 1}, "bogus_field"),
    ({"recency_days": "abc"}, "recency_days")])
def test_invalid_records_get_clear_422(client, patch, field):
    r = client.post("/predict/customer", json={**GOOD, **patch})
    assert r.status_code == 422
    body = r.json()
    assert "Invalid customer record" in body["detail"] and any(field in e["field"] for e in body["errors"])


def test_missing_required_field(client):
    bad = {k: v for k, v in GOOD.items() if k != "total_spend"}
    r = client.post("/predict/customer", json=bad)
    assert r.status_code == 422 and any("total_spend" in e["field"] for e in r.json()["errors"])


def test_batch_scores_valid_and_reports_bad_rows(client):
    rows = pd.DataFrame([{k: v for k, v in GOOD.items() if k != "category_affinity"} | {"affinity_lighting": 0.3}] * 3)
    rows.loc[1, "return_rate"] = 7.0
    buf = io.BytesIO(rows.to_csv(index=False).encode())
    r = client.post("/predict/batch", files={"file": ("b.csv", buf, "text/csv")})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["n_rows"] == 3 and b["n_scored"] == 2 and b["n_failed"] == 1
    assert b["errors"][0]["row"] == 3 and "return_rate" in b["errors"][0]["errors"][0]


def test_batch_rejects_missing_columns_and_empty(client):
    r = client.post("/predict/batch", files={"file": ("b.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert r.status_code == 422 and "missing required columns" in r.json()["detail"]
    r = client.post("/predict/batch", files={"file": ("b.csv", io.BytesIO(b""), "text/csv")})
    assert r.status_code == 422


def test_explain_shap_and_lime(client):
    for method in ("shap", "lime"):
        r = client.post("/explain/customer", json={"record": GOOD, "method": method})
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["method"] == method and b["contributions"] and "churn risk" in b["explanation"].lower() and "Main drivers" in b["explanation"]


def test_persist_prediction(client):
    r = client.post("/predict/customer?persist=true", json={**GOOD, "customer_id": 999999})
    # FK to customers may reject an unknown id on strict engines; SQLite does not enforce FKs by default
    assert r.status_code == 200
    assert client.post("/predict/customer?persist=true", json={**GOOD, "customer_id": None}).status_code == 422


def test_latency_smoke(client):
    client.post("/predict/customer", json=GOOD)
    lat = []
    for _ in range(40):
        t = time.perf_counter()
        client.post("/predict/customer", json=GOOD)
        lat.append((time.perf_counter() - t) * 1000)
    assert np.percentile(lat, 95) < 1000  # loose CI guard; the PRD p95 < 400 ms target is measured in the report benchmark
