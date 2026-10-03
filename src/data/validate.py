"""Automated data validation. Raises DataValidationError (fails loudly) when checks are violated."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataValidationError(ValueError):
    """Raised when a validation check fails."""


def validate_transactions(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Schema, null-rate and date-range checks on the (raw or cleaned) transaction table."""
    dcfg = cfg["data"]
    problems: list[str] = []

    missing = [c for c in dcfg["required_columns"] if c not in df.columns]
    if missing:
        raise DataValidationError(f"Schema check failed, missing columns: {missing}")
    if len(df) == 0:
        raise DataValidationError("Transaction table is empty")

    if not np.issubdtype(df["invoice_date"].dtype, np.datetime64):
        problems.append("invoice_date is not datetime64")
    for col in ("quantity", "unit_price"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"{col} is not numeric")

    null_rates = {c: float(df[c].isna().mean()) for c in dcfg["max_null_rate"]}
    for col, limit in dcfg["max_null_rate"].items():
        if null_rates[col] > limit:
            problems.append(f"null rate for {col} is {null_rates[col]:.3f} > {limit}")

    dmin, dmax = df["invoice_date"].min(), df["invoice_date"].max()
    if dmin < pd.Timestamp(dcfg["expected_date_min"]) or dmax > pd.Timestamp(dcfg["expected_date_max"]):
        problems.append(f"date range {dmin} .. {dmax} outside expected window")

    nonpos = float((df["unit_price"] <= 0).mean())
    if nonpos > dcfg["max_nonpositive_price_rate"]:
        problems.append(f"non-positive price rate {nonpos:.4f} > {dcfg['max_nonpositive_price_rate']}")

    if problems:
        for p in problems:
            logger.error("validation failure: %s", p)
        raise DataValidationError("; ".join(problems))
    logger.info("transaction validation passed (%d rows)", len(df))
    return {"rows": int(len(df)), "null_rates": null_rates, "date_min": str(dmin), "date_max": str(dmax),
            "nonpositive_price_rate": nonpos}


def validate_features(df: pd.DataFrame, required: list[str]) -> None:
    """Validate a customer-level feature table (run before modelling)."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"feature table missing columns: {missing}")
    num = df[required].select_dtypes(include=[np.number])
    if not np.isfinite(num.to_numpy(dtype=float)).all():
        raise DataValidationError("feature table contains NaN or infinite values")
    if (df["recency_days"] < 0).any():
        raise DataValidationError("negative recency_days (point-in-time violation)")
    if (df["frequency"] < 1).any():
        raise DataValidationError("customer with zero orders in feature table")
    if not df["return_rate"].between(0, 1).all():
        raise DataValidationError("return_rate outside [0, 1]")
    if df["customer_id"].duplicated().any():
        raise DataValidationError("duplicate customer_id in feature table")
    logger.info("feature validation passed (%d customers)", len(df))
