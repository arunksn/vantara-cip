"""FastAPI application: ``uvicorn api.main:app``.

Interactive OpenAPI documentation is served at /docs (Swagger UI) and /redoc.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.routers import explain, meta, predict
from src.models.predictor import Predictor
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger("api")


def create_app(config_path: str | None = None, database_url: str | None = None) -> FastAPI:
    """Application factory (used by uvicorn and by tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        cfg = load_config(config_path)
        app.state.predictor = Predictor(cfg)
        url = database_url or os.environ.get("DATABASE_URL")
        app.state.engine = None
        if url:
            from src.utils.db import get_engine, init_db

            try:
                app.state.engine = get_engine(url)
                init_db(app.state.engine)
            except Exception as exc:  # noqa: BLE001
                logger.warning("database unavailable at startup: %s", exc)
        logger.info("API ready: model %s", app.state.predictor.meta.get("production_model"))
        yield

    app = FastAPI(title="Customer Behavior Prediction API", version="1.0.0", lifespan=lifespan,
                  description="Churn probability, 90-day CLV, anomaly flag and explanations per customer.")

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"field": ".".join(str(p) for p in e["loc"][1:]) or str(e["loc"][0]), "message": e["msg"]}
                  for e in exc.errors()]
        return JSONResponse(status_code=422, content=jsonable_encoder(
            {"detail": "Invalid customer record. Fix the listed fields and retry.", "errors": errors}))

    app.include_router(meta.router)
    app.include_router(predict.router)
    app.include_router(explain.router)
    return app


app = create_app()
