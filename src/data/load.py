"""Data collection: load both sheets of Online Retail II, standardise, and order chronologically.

Option B (ucimlrepo) is the default; Option A (manual xlsx in data/raw/) is the fallback.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import path_of
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_ALIASES = {
    "invoice": "invoice",
    "invoiceno": "invoice",
    "stockcode": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "price": "unit_price",
    "unitprice": "unit_price",
    "customerid": "customer_id",
    "country": "country",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename source columns to canonical snake_case names and enforce dtypes."""
    renamed = {}
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]", "", str(col).lower())
        if key in _ALIASES:
            renamed[col] = _ALIASES[key]
    out = df.rename(columns=renamed)
    out = out.loc[:, ~out.columns.duplicated()]
    keep = [c for c in _ALIASES.values() if c in out.columns]
    out = out[list(dict.fromkeys(keep))].copy()

    out["invoice"] = out["invoice"].astype(str).str.strip()
    out["stock_code"] = out["stock_code"].astype(str).str.strip().str.upper()
    out["description"] = out["description"].astype("string")
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["invoice_date"] = pd.to_datetime(out["invoice_date"], errors="coerce")
    out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce")
    out["customer_id"] = pd.to_numeric(out["customer_id"], errors="coerce").astype("Int64")
    out["country"] = out["country"].astype("string").str.strip()
    return out


def combine_sheets(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate yearly sheets into one chronologically ordered table."""
    df = pd.concat([standardize_columns(f) for f in frames], ignore_index=True)
    return df.sort_values("invoice_date", kind="stable").reset_index(drop=True)


def _load_file(path: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith((".csv", ".csv.gz")):
        return combine_sheets([pd.read_csv(path)])
    sheets = cfg["data"]["sheets"]
    return combine_sheets([pd.read_excel(path, sheet_name=s) for s in sheets])


def _load_ucimlrepo(cfg: dict[str, Any]) -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo  # imported lazily: optional in offline environments

    ds = fetch_ucirepo(id=cfg["data"]["uci_id"])
    parts = [p for p in (ds.data.ids, ds.data.features) if p is not None and len(p.columns)]
    return combine_sheets([pd.concat(parts, axis=1)])


def load_raw(cfg: dict[str, Any], input_path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw transaction table.

    Order: explicit input_path, cached ucimlrepo download, Option B (ucimlrepo), Option A (xlsx).
    """
    if input_path is not None:
        logger.info("loading raw data from explicit path: %s", input_path)
        return _load_file(Path(input_path), cfg)

    cache = path_of(cfg, "raw_cache")
    if cache.exists():
        logger.info("loading cached ucimlrepo download: %s", cache)
        return _load_file(cache, cfg)

    try:
        logger.info("fetching Online Retail II via ucimlrepo (Option B)")
        df = _load_ucimlrepo(cfg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        return df
    except Exception as exc:  # noqa: BLE001 - any failure triggers the documented fallback
        logger.warning("ucimlrepo unavailable (%s); falling back to Option A xlsx", exc)

    xlsx = path_of(cfg, "raw_xlsx")
    if xlsx.exists():
        return _load_file(xlsx, cfg)
    raise FileNotFoundError(
        f"Could not obtain the dataset. Install ucimlrepo with network access (Option B) or place "
        f"online_retail_II.xlsx at {xlsx} (Option A, see README)."
    )
