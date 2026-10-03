"""Explanation endpoint (SHAP or LIME) for one customer record."""
from __future__ import annotations

from fastapi import APIRouter, Request

from api.routers.predict import record_to_frame
from api.schemas.customer import ExplainRequest
from api.schemas.responses import ExplanationResponse

router = APIRouter(prefix="/explain", tags=["explainability"])


@router.post("/customer", response_model=ExplanationResponse)
def explain_customer(req: ExplainRequest, request: Request) -> ExplanationResponse:
    """Per-feature contributions and a plain-language explanation for one customer."""
    out = request.app.state.predictor.explain(record_to_frame(req.record), req.method)
    return ExplanationResponse(**out)
