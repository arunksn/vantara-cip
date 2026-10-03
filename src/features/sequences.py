"""Time-ordered purchase sequences for the LSTM (point-in-time safe).

Each customer becomes up to ``max_len`` order events followed by one terminal "cutoff marker" step whose gap is
the number of days from the last order to the prediction date. Per step:
  numeric = [log1p(order value) / 10, log1p(gap days) / 6, is_event]
  category = dominant product category of the order (1..K+1); 0 = padding / cutoff marker.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.builders import point_in_time_frame


def category_vocab(cfg: dict[str, Any]) -> list[str]:
    """Category vocabulary in index order (index 0 reserved for padding)."""
    return ["<pad>"] + list(cfg["features"]["category_rules"]) + ["other"]


def build_sequences(
    tx: pd.DataFrame, customer_ids: np.ndarray, cutoff: pd.Timestamp, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (numeric array [n, T+1, 3], category array [n, T+1]) aligned to customer_ids."""
    max_len = int(cfg["features"]["sequence_max_len"])
    vocab = {c: i for i, c in enumerate(category_vocab(cfg))}
    sales = point_in_time_frame(tx, cutoff)
    sales = sales[~sales["is_return"]]

    cat_spend = sales.groupby(["customer_id", "invoice", "category"])["line_total"].sum().reset_index()
    dom = cat_spend.sort_values("line_total", ascending=False).drop_duplicates(["customer_id", "invoice"])
    orders = (
        sales.groupby(["customer_id", "invoice"]).agg(order_date=("invoice_date", "max"),
                                                     value=("line_total", "sum")).reset_index()
        .merge(dom[["customer_id", "invoice", "category"]], on=["customer_id", "invoice"], how="left")
        .sort_values(["customer_id", "order_date"], kind="stable")
    )
    orders["gap"] = orders.groupby("customer_id")["order_date"].diff().dt.total_seconds().div(86400).fillna(0.0)
    orders["cat_idx"] = orders["category"].map(vocab).fillna(vocab["other"]).astype(int)
    orders["amt"] = np.log1p(orders["value"].clip(lower=0)) / 10.0
    orders["gap_s"] = np.log1p(orders["gap"].clip(lower=0)) / 6.0

    n = len(customer_ids)
    num = np.zeros((n, max_len + 1, 3), dtype="float32")
    cat = np.zeros((n, max_len + 1), dtype="int32")
    pos = {cid: i for i, cid in enumerate(customer_ids)}
    for cid, grp in orders.groupby("customer_id", sort=False):
        i = pos.get(cid)
        if i is None:
            continue
        g = grp.tail(max_len)
        k = len(g)
        start = max_len - k
        num[i, start:max_len, 0] = g["amt"].to_numpy()
        num[i, start:max_len, 1] = g["gap_s"].to_numpy()
        num[i, start:max_len, 2] = 1.0
        cat[i, start:max_len] = g["cat_idx"].to_numpy()
        days_since = (cutoff - grp["order_date"].iloc[-1]).total_seconds() / 86400.0
        num[i, max_len, 1] = np.log1p(max(days_since, 0.0)) / 6.0
    return num, cat
