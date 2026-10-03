"""Health-check and model-metadata endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import text

from api.schemas.responses import HealthResponse

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness/readiness: model loaded and database reachable (if configured)."""
    db = "not_configured"
    engine = request.app.state.engine
    if engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db = "ok"
        except Exception:  # noqa: BLE001
            db = "unreachable"
    loaded = getattr(request.app.state, "predictor", None) is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded, database=db)


@router.get("/model/metadata")
def model_metadata(request: Request) -> dict:
    """Model version, training date, features, operating threshold and headline metrics."""
    m = request.app.state.predictor.meta
    keys = ["run_id", "trained_at", "production_model", "churn_threshold", "model_features", "cutoffs", "data_source",
            "production_cv_auc", "versions", "seed", "global_importance", "vif_dropped"]
    out = {k: m.get(k) for k in keys}
    out["test_metrics"] = m.get("production_test_metrics")
    return out
