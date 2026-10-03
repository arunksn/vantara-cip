"""Pydantic request models with explicit range validation and clear error messages."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerRecord(BaseModel):
    """Engineered customer features as of the prediction date (same definitions as the training pipeline)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "customer_id": 12346, "country": "United Kingdom", "recency_days": 45, "frequency": 6, "total_spend": 1450.5,
        "avg_spend": 241.75, "historical_clv": 1380.0, "avg_basket_size": 120, "freq_trend": -0.1, "gap_variance": 900,
        "seasonal_concentration": 0.2, "return_rate": 0.02, "discount_sensitivity": 0.1, "avg_product_popularity": 0.05,
        "category_affinity": {"lighting": 0.3, "kitchen_dining": 0.2}}})

    customer_id: int | None = Field(None, description="Optional identifier (needed only to persist the prediction)")
    country: str = Field(..., min_length=1, max_length=64)
    recency_days: float = Field(..., ge=0, description="Days since last purchase")
    frequency: float = Field(..., ge=1, description="Number of distinct orders")
    total_spend: float = Field(..., ge=0)
    avg_spend: float = Field(..., ge=0)
    historical_clv: float = Field(..., ge=0, description="Net revenue to date (sales minus returns)")
    avg_basket_size: float = Field(..., ge=0)
    freq_trend: float = Field(0.0, description="Slope of orders per 30-day bin")
    gap_variance: float = Field(0.0, ge=0, description="Variance of days between orders")
    seasonal_concentration: float = Field(..., gt=0, le=1, description="Herfindahl index of monthly order shares")
    return_rate: float = Field(..., ge=0, le=1)
    discount_sensitivity: float = Field(..., ge=0, le=1)
    avg_product_popularity: float = Field(..., ge=0, le=1)
    engagement_score: float | None = Field(None, ge=0, le=100, description="Computed server-side if omitted")
    category_affinity: dict[str, float] = Field(default_factory=dict, description="Share of spend per category")

    @field_validator("category_affinity")
    @classmethod
    def _affinity_ok(cls, v: dict[str, float]) -> dict[str, float]:
        for k, val in v.items():
            if not 0 <= val <= 1:
                raise ValueError(f"category_affinity['{k}'] must be between 0 and 1, got {val}")
        if sum(v.values()) > 1.0 + 1e-6:
            raise ValueError("category_affinity values must sum to at most 1")
        return v


class ExplainRequest(BaseModel):
    """Explanation request for one customer."""

    record: CustomerRecord
    method: Literal["shap", "lime"] = "shap"
