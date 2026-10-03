"""Data cleaning: versioned, unit-testable pandas functions shared by training and scoring.

Cleaning rules (documented, see README):
  * Missing Customer ID rows are FLAGGED (has_customer_id) and kept for product-level analysis.
  * Exact duplicate lines are removed; repeated purchases of the same SKU in one order with
    different quantities/prices are preserved.
  * Negative quantities (invoice starting with "C" or quantity < 0) are flagged as returns, not dropped.
  * Rows with price <= 0 (adjustments / manual corrections) are flagged (is_price_adjustment) and
    excluded from customer-level modelling.
  * Quantity/price outliers use IQR bounds; an outlier that is exactly reversed by an opposite-sign line
    for the same customer and SKU (or beyond an absolute ceiling) is a data-entry error and is excluded;
    other outliers are legitimate bulk purchases and kept (flagged).
  * Descriptions are standardised through a StockCode-keyed lookup table.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

VERSION = "1.0.0"


def normalize_description(s: pd.Series) -> pd.Series:
    """Upper-case, strip and collapse whitespace."""
    return (
        s.astype("string").str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    )


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows identical on every column (keeps first occurrence)."""
    before = len(df)
    out = df.drop_duplicates(keep="first").reset_index(drop=True)
    return out, before - len(out)


def flag_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Flag cancellations / returns explicitly (never dropped: they are a predictive signal)."""
    out = df.copy()
    out["is_return"] = out["invoice"].str.upper().str.startswith("C") | (out["quantity"] < 0)
    return out


def flag_price_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """Flag zero/negative price rows (adjustments, manual corrections)."""
    out = df.copy()
    out["is_price_adjustment"] = out["unit_price"] <= 0
    return out


def flag_admin_codes(df: pd.DataFrame, regex: str) -> pd.DataFrame:
    """Flag non-product StockCodes (postage, bank charges, manual entries, ...)."""
    out = df.copy()
    pattern = re.compile(regex, flags=re.IGNORECASE)
    out["is_admin_code"] = out["stock_code"].map(lambda x: bool(pattern.match(str(x))))
    return out


def iqr_bounds(values: pd.Series, k: float) -> tuple[float, float]:
    """Tukey fences Q1 - k*IQR, Q3 + k*IQR."""
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def flag_outliers(df: pd.DataFrame, k: float, abs_qty_ceiling: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Flag IQR outliers and separate data-entry errors from legitimate bulk purchases."""
    out = df.copy()
    valid = ~out["is_price_adjustment"]
    q_lo, q_hi = iqr_bounds(out.loc[valid, "quantity"].abs(), k)
    p_lo, p_hi = iqr_bounds(out.loc[valid, "unit_price"], k)
    qty_outlier = valid & (out["quantity"].abs() > q_hi)
    out["is_outlier"] = qty_outlier | (valid & (out["unit_price"] > p_hi))

    out["is_entry_error"] = False
    # Reversal pairing only considers QUANTITY outliers: a fully returned expensive item is legitimate.
    cand = out.loc[qty_outlier, ["customer_id", "stock_code", "quantity"]].copy()
    cand["abs_q"] = cand["quantity"].abs()
    cand["sign"] = np.sign(cand["quantity"])
    cand = cand.dropna(subset=["customer_id"])
    if len(cand):
        signs = cand.groupby(["customer_id", "stock_code", "abs_q"])["sign"].agg(["min", "max"])
        reversed_keys = signs[(signs["min"] < 0) & (signs["max"] > 0)].index
        key_index = pd.MultiIndex.from_frame(cand[["customer_id", "stock_code", "abs_q"]])
        paired = key_index.isin(reversed_keys)
        out.loc[cand.index[paired], "is_entry_error"] = True
    out.loc[qty_outlier & (out["quantity"].abs() >= abs_qty_ceiling), "is_entry_error"] = True
    out["is_bulk_outlier"] = out["is_outlier"] & ~out["is_entry_error"]
    info = {"quantity_bounds_abs": [q_lo, q_hi], "price_bounds": [p_lo, p_hi],
            "n_outliers": int(out["is_outlier"].sum()),
            "n_entry_errors": int(out["is_entry_error"].sum()),
            "n_bulk_outliers": int(out["is_bulk_outlier"].sum())}
    return out, info


def build_description_lookup(df: pd.DataFrame, typos: dict[str, str] | None = None) -> pd.Series:
    """StockCode -> most frequent normalised description (after known-typo fixes)."""
    desc = normalize_description(df["description"])
    if typos:
        mapping = {k.upper(): v.upper() for k, v in typos.items()}
        desc = desc.replace(mapping)
    tmp = pd.DataFrame({"stock_code": df["stock_code"], "desc": desc}).dropna()
    counts = tmp.groupby(["stock_code", "desc"]).size().reset_index(name="n")
    counts = counts.sort_values(["stock_code", "n", "desc"], ascending=[True, False, True])
    return counts.drop_duplicates("stock_code").set_index("stock_code")["desc"]


def standardize_descriptions(df: pd.DataFrame, typos: dict[str, str] | None = None) -> pd.DataFrame:
    """Replace descriptions with the StockCode-keyed canonical description."""
    out = df.copy()
    lookup = build_description_lookup(out, typos)
    out["description"] = out["stock_code"].map(lookup).fillna(normalize_description(out["description"]))
    return out


def assign_category(descriptions: pd.Series, rules: dict[str, list[str]]) -> pd.Series:
    """Map descriptions to product categories with first-match keyword rules (else 'other')."""
    uniq = descriptions.dropna().unique()

    def _cat(text: str) -> str:
        up = str(text).upper()
        for cat, words in rules.items():
            for w in words:
                if re.search(rf"\b{re.escape(w.upper())}S?\b", up):
                    return cat
        return "other"

    mapping = {u: _cat(u) for u in uniq}
    return descriptions.map(mapping).fillna("other").astype(str)


def clean_transactions(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full cleaning chain. Returns the cleaned table and a cleaning report."""
    dcfg = cfg["data"]
    report: dict[str, Any] = {"cleaning_version": VERSION, "rows_in": int(len(df))}
    df = df.dropna(subset=["invoice_date", "quantity", "unit_price"]).copy()
    df, n_dup = drop_exact_duplicates(df)
    report["exact_duplicates_removed"] = int(n_dup)

    df["has_customer_id"] = df["customer_id"].notna()
    report["rows_missing_customer_id"] = int((~df["has_customer_id"]).sum())

    df = flag_returns(df)
    df = flag_price_adjustments(df)
    df = flag_admin_codes(df, dcfg["admin_stockcode_regex"])
    df, out_info = flag_outliers(df, dcfg["iqr_multiplier"], dcfg["entry_error_abs_quantity"])
    report["outliers"] = out_info
    df = standardize_descriptions(df, dcfg.get("known_description_typos") or {})
    df["category"] = assign_category(df["description"], cfg["features"]["category_rules"])
    df["line_total"] = df["quantity"] * df["unit_price"]
    df["use_for_customer_model"] = df["has_customer_id"] & ~df["is_price_adjustment"] & ~df["is_entry_error"]

    report.update({
        "rows_out": int(len(df)),
        "return_lines": int(df["is_return"].sum()),
        "price_adjustment_lines": int(df["is_price_adjustment"].sum()),
        "admin_code_lines": int(df["is_admin_code"].sum()),
        "rows_used_for_customer_model": int(df["use_for_customer_model"].sum()),
        "unique_customers": int(df.loc[df["has_customer_id"], "customer_id"].nunique()),
    })
    df = df.sort_values("invoice_date", kind="stable").reset_index(drop=True)
    logger.info("cleaning complete: %s", report)
    return df, report
