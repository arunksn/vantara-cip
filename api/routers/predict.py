"""Prediction endpoints: single customer and CSV batch."""
from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from pydantic import ValidationError

from api.schemas.customer import CustomerRecord
from api.schemas.responses import BatchResponse, PredictionResponse, RowError
from src.models.predictor import Predictor
from src.utils.logging_utils import get_logger

router = APIRouter(prefix="/predict", tags=["prediction"])
logger = get_logger(__name__)


def record_to_frame(rec: CustomerRecord) -> pd.DataFrame:
    """CustomerRecord -> one-row frame with flattened affinity columns."""
    d = rec.model_dump(exclude={"category_affinity"})
    for cat, val in rec.category_affinity.items():
        d[f"affinity_{cat}"] = val
    return pd.DataFrame([d])


def _to_response(pred: pd.Series, customer_id: int | None, version: str) -> PredictionResponse:
    return PredictionResponse(
        customer_id=customer_id, churn_probability=float(pred["churn_probability"]),
        churn_predicted=bool(pred["churn_predicted"]), risk_tier=str(pred["risk_tier"]),
        next_purchase_probability=float(pred["next_purchase_probability"]),
        predicted_clv_90d=float(pred["predicted_clv_90d"]), anomaly_score=float(pred["anomaly_score"]),
        is_anomaly=bool(pred["is_anomaly"]), recommended_categories=list(pred["recommended_categories"]),
        model_version=version)


@router.post("/customer", response_model=PredictionResponse)
def predict_customer(rec: CustomerRecord, request: Request, persist: bool = Query(False)) -> PredictionResponse:
    """Churn probability, risk tier, 90-day CLV, anomaly flag and recommended categories for one customer."""
    predictor: Predictor = request.app.state.predictor
    frame = record_to_frame(rec)
    pred = predictor.predict(frame).iloc[0]
    resp = _to_response(pred, rec.customer_id, predictor.meta["run_id"])
    if persist:
        engine = request.app.state.engine
        if engine is None or rec.customer_id is None:
            raise HTTPException(422, "persist=true requires customer_id and a configured database")
        from datetime import datetime, timezone

        from src.utils.db import upsert_prediction

        upsert_prediction(engine, {
            "customer_id": rec.customer_id, "churn_probability": resp.churn_probability,
            "churn_predicted": int(resp.churn_predicted), "risk_tier": resp.risk_tier,
            "predicted_clv_90d": resp.predicted_clv_90d, "at_risk_value": resp.churn_probability * rec.historical_clv,
            "anomaly_score": resp.anomaly_score, "is_anomaly": resp.is_anomaly,
            "recommended_categories": ",".join(resp.recommended_categories), "top_reasons": None, "shap_json": None,
            "model_version": resp.model_version, "scored_at": datetime.now(timezone.utc).replace(tzinfo=None)})
    return resp


@router.post("/batch", response_model=BatchResponse)
async def predict_batch(request: Request, file: UploadFile = File(..., description="CSV of customer feature rows")) -> BatchResponse:
    """Score a CSV upload. Invalid rows are reported per row; valid rows are still scored."""
    predictor: Predictor = request.app.state.predictor
    max_rows = predictor.cfg["serving"]["max_batch_rows"]
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"Could not parse CSV: {exc}") from exc
    if df.empty:
        raise HTTPException(422, "CSV contains no rows")
    if len(df) > max_rows:
        raise HTTPException(413, f"CSV has {len(df)} rows; the limit is {max_rows}")
    required = [n for n, f in CustomerRecord.model_fields.items() if f.is_required()]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(422, f"CSV is missing required columns: {missing}")

    aff_cols = [c for c in df.columns if c.startswith("affinity_")]
    valid_rows, valid_idx, errors = [], [], []
    for i, row in df.iterrows():
        data = {k: v for k, v in row.to_dict().items() if k in CustomerRecord.model_fields and pd.notna(v)}
        data["category_affinity"] = {c[9:]: float(row[c]) for c in aff_cols if pd.notna(row[c])}
        try:
            valid_rows.append(record_to_frame(CustomerRecord(**data)).iloc[0])
            valid_idx.append(i)
        except ValidationError as exc:
            errors.append(RowError(row=int(i) + 2, errors=[f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]))
    preds: list[PredictionResponse] = []
    if valid_rows:
        frame = pd.DataFrame(valid_rows).reset_index(drop=True)
        out = predictor.predict(frame)
        ids = frame["customer_id"] if "customer_id" in frame else pd.Series([None] * len(frame))
        preds = [_to_response(out.iloc[j], None if pd.isna(ids.iloc[j]) else int(ids.iloc[j]), predictor.meta["run_id"])
                 for j in range(len(out))]
    return BatchResponse(n_rows=len(df), n_scored=len(preds), n_failed=len(errors), errors=errors, predictions=preds)
