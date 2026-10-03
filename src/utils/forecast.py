"""Simple revenue forecast for the dashboard: seasonal-naive x year-over-year growth, else linear trend."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drop_partial_last_month(monthly: pd.DataFrame) -> pd.DataFrame:
    """Drop the final month if the data stops before that month ends (it would bias the trend)."""
    last = monthly.iloc[-1]
    month_end = (last["month"] + pd.offsets.MonthEnd(0)).normalize()
    if last["last_date"].normalize() < month_end - pd.Timedelta(days=2):
        return monthly.iloc[:-1].reset_index(drop=True)
    return monthly


def forecast_revenue(monthly: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Forecast the next ``horizon`` months. Expects columns month, revenue (complete months only)."""
    m = monthly.sort_values("month").reset_index(drop=True)
    rev = m["revenue"].to_numpy(float)
    future = [m["month"].iloc[-1] + pd.offsets.MonthBegin(i) for i in range(1, horizon + 1)]
    if len(rev) >= 24:
        growth = rev[-12:].sum() / max(rev[-24:-12].sum(), 1e-9)
        vals = [rev[len(rev) - 12 + i] * growth for i in range(horizon)]
        method = "seasonal-naive x YoY growth"
    else:
        x = np.arange(len(rev))
        slope, intercept = np.polyfit(x, rev, 1) if len(rev) > 1 else (0.0, rev[-1])
        vals = [max(slope * (len(rev) + i) + intercept, 0.0) for i in range(horizon)]
        method = "linear trend"
    return pd.DataFrame({"month": future, "revenue": vals, "method": method})
