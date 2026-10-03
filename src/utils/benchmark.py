"""Latency benchmarks: API p95 (in-process TestClient, server-side) and dashboard script run time (Streamlit AppTest)."""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import artifacts_path, path_of
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def benchmark_api(cfg: dict[str, Any]) -> dict[str, Any]:
    """Measure single-customer prediction latency (p50/p95/p99) over N requests."""
    from fastapi.testclient import TestClient

    from api.main import create_app
    from api.routers.predict import record_to_frame  # noqa: F401

    feats = pd.read_csv(path_of(cfg, "features_scoring")).sample(
        cfg["serving"]["api_benchmark_requests"], replace=True, random_state=cfg["project"]["seed"])
    aff = [c for c in feats.columns if c.startswith("affinity_")]
    payloads = []
    for _, r in feats.iterrows():
        payloads.append({
            "customer_id": int(r["customer_id"]), "country": str(r["country"]),
            **{k: float(r[k]) for k in ["recency_days", "frequency", "total_spend", "avg_spend", "historical_clv",
                                         "avg_basket_size", "freq_trend", "gap_variance", "seasonal_concentration",
                                         "return_rate", "discount_sensitivity", "avg_product_popularity"]},
            "category_affinity": {c[9:]: float(r[c]) for c in aff if r[c] > 0}})
    lat = []
    with TestClient(create_app()) as client:
        for p in payloads[:5]:
            client.post("/predict/customer", json=p)  # warm-up
        for p in payloads:
            t = time.perf_counter()
            resp = client.post("/predict/customer", json=p)
            lat.append((time.perf_counter() - t) * 1000)
            assert resp.status_code == 200, resp.text
    res = {"requests": len(lat), "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
           "p99_ms": float(np.percentile(lat, 99)), "method": "in-process TestClient, single worker (excludes network)"}
    logger.info("API latency: %s", res)
    return res


def benchmark_dashboard(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Server-side script run time of the dashboard with a local SQLite database built from the processed files."""
    import os
    import tempfile
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from src.models.batch_score import load_database
    from src.utils.db import get_engine  # noqa: F401

    root = Path(cfg["_root"])
    dash = root / "frontend" / "dashboard.py"
    if not dash.exists():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        url = f"sqlite:///{tmp}/bench.db"
        old = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = url
        try:
            load_database(cfg)
            at = AppTest.from_file(str(dash), default_timeout=60)
            t0 = time.perf_counter()
            at.run()
            first = time.perf_counter() - t0
            t1 = time.perf_counter()
            at.run()
            second = time.perf_counter() - t1
            exc = [str(e.value) for e in at.exception]
        finally:
            if old is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = old
    res = {"first_load_sec": first, "cached_load_sec": second, "exceptions": exc,
           "method": "Streamlit AppTest server-side script run (excludes browser rendering)"}
    logger.info("dashboard load: %s", res)
    return res


def run_benchmarks(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run both benchmarks and persist results."""
    out: dict[str, Any] = {}
    for name, fn in (("api", benchmark_api), ("dashboard", benchmark_dashboard)):
        try:
            out[name] = fn(cfg)
        except Exception as exc:  # noqa: BLE001 - benchmarks must not break the pipeline
            logger.warning("%s benchmark failed: %s", name, exc)
            out[name] = {"error": str(exc)}
    artifacts_path(cfg, "metrics", "benchmarks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
