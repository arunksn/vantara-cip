"""Response models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    customer_id: int | None = None
    churn_probability: float
    churn_predicted: bool
    risk_tier: str
    next_purchase_probability: float
    predicted_clv_90d: float
    anomaly_score: float
    is_anomaly: bool
    recommended_categories: list[str]
    model_version: str


class RowError(BaseModel):
    row: int
    errors: list[str]


class BatchResponse(BaseModel):
    n_rows: int
    n_scored: int
    n_failed: int
    errors: list[RowError]
    predictions: list[PredictionResponse]


class ExplanationResponse(BaseModel):
    method: str
    churn_probability: float
    explanation: str
    contributions: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database: str
