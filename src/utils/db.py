"""PostgreSQL persistence (SQLAlchemy). Tables: customers, transactions, predictions, segments.

The connection string comes from DATABASE_URL (never hard-coded). SQLite URLs are accepted for local tests.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base."""


class Customer(Base):
    __tablename__ = "customers"
    customer_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    country: Mapped[str] = mapped_column(String(64))
    first_purchase: Mapped[Any] = mapped_column(DateTime)
    last_purchase: Mapped[Any] = mapped_column(DateTime)
    recency_days: Mapped[float] = mapped_column(Float)
    frequency: Mapped[float] = mapped_column(Float)
    total_spend: Mapped[float] = mapped_column(Float)
    historical_clv: Mapped[float] = mapped_column(Float)
    engagement_score: Mapped[float] = mapped_column(Float)
    features_json: Mapped[str] = mapped_column(Text)


class Transaction(Base):
    __tablename__ = "transactions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice: Mapped[str] = mapped_column(String(32))
    stock_code: Mapped[str] = mapped_column(String(32))
    description: Mapped[str] = mapped_column(String(255), nullable=True)
    category: Mapped[str] = mapped_column(String(32))
    quantity: Mapped[int] = mapped_column(Integer)
    invoice_date: Mapped[Any] = mapped_column(DateTime, index=True)
    unit_price: Mapped[float] = mapped_column(Float)
    line_total: Mapped[float] = mapped_column(Float)
    is_return: Mapped[bool] = mapped_column(Boolean)
    customer_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("customers.customer_id"), index=True)
    country: Mapped[str] = mapped_column(String(64))


class Prediction(Base):
    __tablename__ = "predictions"
    customer_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("customers.customer_id"), primary_key=True)
    churn_probability: Mapped[float] = mapped_column(Float)
    churn_predicted: Mapped[int] = mapped_column(Integer)
    risk_tier: Mapped[str] = mapped_column(String(8))
    predicted_clv_90d: Mapped[float] = mapped_column(Float)
    at_risk_value: Mapped[float] = mapped_column(Float)
    anomaly_score: Mapped[float] = mapped_column(Float)
    is_anomaly: Mapped[bool] = mapped_column(Boolean)
    recommended_categories: Mapped[str] = mapped_column(String(128))
    top_reasons: Mapped[str] = mapped_column(Text, nullable=True)
    shap_json: Mapped[str] = mapped_column(Text, nullable=True)
    model_version: Mapped[str] = mapped_column(String(64))
    scored_at: Mapped[Any] = mapped_column(DateTime)


class Segment(Base):
    __tablename__ = "segments"
    customer_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("customers.customer_id"), primary_key=True)
    kmeans_segment: Mapped[int] = mapped_column(Integer)
    segment_label: Mapped[str] = mapped_column(String(64))
    secondary_cluster: Mapped[int] = mapped_column(Integer)
    value_tier: Mapped[str] = mapped_column(String(8))
    pca_x: Mapped[float] = mapped_column(Float)
    pca_y: Mapped[float] = mapped_column(Float)


def get_engine(url: str | None = None) -> Engine:
    """Create an engine from an explicit URL or the DATABASE_URL environment variable."""
    url = url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (see .env.example)")
    return create_engine(url, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    """Create tables if missing."""
    Base.metadata.create_all(engine)


def replace_table(engine: Engine, model: type[Base], df: pd.DataFrame, chunksize: int = 10000) -> int:
    """Delete all rows of a table then bulk-insert df (idempotent reload)."""
    cols = [c.name for c in model.__table__.columns if c.name in df.columns]
    with engine.begin() as conn:
        conn.execute(delete(model))
    df[cols].to_sql(model.__tablename__, engine, if_exists="append", index=False, chunksize=chunksize, method="multi")
    return len(df)


def load_all(engine: Engine, customers: pd.DataFrame, transactions: pd.DataFrame | None,
             predictions: pd.DataFrame, segments: pd.DataFrame) -> dict[str, int]:
    """Reload every table in dependency order (children cleared first)."""
    init_db(engine)
    with engine.begin() as conn:
        for m in (Transaction, Prediction, Segment):
            conn.execute(delete(m))
        conn.execute(delete(Customer))
    counts = {"customers": replace_table(engine, Customer, customers)}
    if transactions is not None:
        counts["transactions"] = replace_table(engine, Transaction, transactions)
    counts["predictions"] = replace_table(engine, Prediction, predictions)
    counts["segments"] = replace_table(engine, Segment, segments)
    logger.info("database loaded: %s", counts)
    return counts


def upsert_prediction(engine: Engine, row: dict[str, Any]) -> None:
    """Insert or replace a single prediction row (used by the API when persist=true)."""
    with engine.begin() as conn:
        conn.execute(delete(Prediction).where(Prediction.customer_id == row["customer_id"]))
        conn.execute(Prediction.__table__.insert().values(**row))


def read_sql(engine: Engine, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run a read-only query and return a DataFrame."""
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def monthly_revenue(engine: Engine) -> pd.DataFrame:
    """Net revenue per calendar month (dialect-aware aggregation)."""
    if engine.dialect.name == "postgresql":
        month = "date_trunc('month', invoice_date)"
    else:
        month = "strftime('%Y-%m-01', invoice_date)"
    df = read_sql(engine, f"SELECT {month} AS month, SUM(line_total) AS revenue, MAX(invoice_date) AS last_date "
                          f"FROM transactions GROUP BY 1 ORDER BY 1")
    df["month"] = pd.to_datetime(df["month"])
    df["last_date"] = pd.to_datetime(df["last_date"])
    return df
