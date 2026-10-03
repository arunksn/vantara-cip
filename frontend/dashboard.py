"""Streamlit stakeholder dashboard: ``streamlit run frontend/dashboard.py``.

Reads stored predictions/segments from PostgreSQL (DATABASE_URL) and calls the API (API_URL) for batch scoring
and on-demand SHAP/LIME explanations. Business users never need to run any code.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.db import get_engine, monthly_revenue, read_sql  # noqa: E402
from src.utils.forecast import drop_partial_last_month, forecast_revenue  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Customer Behavior Dashboard", layout="wide")


@st.cache_resource
def engine():
    return get_engine()


@st.cache_data(ttl=600)
def load_main() -> pd.DataFrame:
    q = """SELECT c.customer_id, c.country, c.recency_days, c.frequency, c.total_spend, c.historical_clv,
                  c.engagement_score, c.features_json, p.churn_probability, p.risk_tier, p.predicted_clv_90d,
                  p.at_risk_value, p.is_anomaly, p.recommended_categories, p.top_reasons, p.shap_json,
                  s.segment_label, s.value_tier, s.pca_x, s.pca_y
           FROM customers c JOIN predictions p ON p.customer_id = c.customer_id
           JOIN segments s ON s.customer_id = c.customer_id"""
    return read_sql(engine(), q)


@st.cache_data(ttl=600)
def load_revenue() -> pd.DataFrame:
    return monthly_revenue(engine())


def api_post(path: str, **kwargs):
    r = requests.post(f"{API_URL}{path}", timeout=120, **kwargs)
    return r


st.title("Customer Behavior Prediction Dashboard")
try:
    df = load_main()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read the database ({exc}). Check DATABASE_URL and that the init job has run.")
    st.stop()
if df.empty:
    st.warning("No scored customers found yet. Run the batch scoring job (python -m src.models.batch_score --load-db).")
    st.stop()

with st.sidebar:
    st.header("Filters")
    segs = st.multiselect("Segment", sorted(df["segment_label"].unique()))
    countries = st.multiselect("Country", sorted(df["country"].unique()))
    tiers = st.multiselect("Value tier", ["High", "Medium", "Low"])
    st.caption("Filters apply to the Segments and Churn tabs.")
f = df.copy()
if segs:
    f = f[f["segment_label"].isin(segs)]
if countries:
    f = f[f["country"].isin(countries)]
if tiers:
    f = f[f["value_tier"].isin(tiers)]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(f):,}")
k2.metric("High churn risk", f"{(f['risk_tier'] == 'High').sum():,}")
k3.metric("Revenue at risk (GBP)", f"{f.loc[f['risk_tier'] == 'High', 'at_risk_value'].sum():,.0f}")
k4.metric("Anomalous accounts", f"{int(f['is_anomaly'].sum()):,}")

tab_seg, tab_churn, tab_rev, tab_explain, tab_batch = st.tabs(
    ["Customer segments", "Churn leaderboard", "Revenue trend", "Explain a customer", "Score a CSV"])

with tab_seg:
    st.caption("Each dot is a customer; colours are behavioural segments. Positions are a 2-D projection of their behaviour.")
    st.plotly_chart(px.scatter(f, x="pca_x", y="pca_y", color="segment_label", hover_data=["customer_id", "country"],
                               opacity=0.7, height=480), use_container_width=True)
    prof = f.groupby("segment_label").agg(customers=("customer_id", "count"), avg_recency_days=("recency_days", "mean"),
                                          avg_orders=("frequency", "mean"), avg_net_revenue=("historical_clv", "mean"),
                                          avg_churn_risk=("churn_probability", "mean")).round(2).reset_index()
    st.dataframe(prof, use_container_width=True, hide_index=True)

with tab_churn:
    st.caption("Sorted by revenue at risk = churn probability x net revenue to date, so high-value, high-risk customers come first.")
    n = st.slider("Rows", 10, 200, 50)
    board = f.sort_values("at_risk_value", ascending=False).head(n)[
        ["customer_id", "country", "segment_label", "risk_tier", "churn_probability", "historical_clv", "at_risk_value",
         "recency_days", "recommended_categories"]]
    st.dataframe(board, use_container_width=True, hide_index=True)
    st.download_button("Download leaderboard (CSV)", board.to_csv(index=False), "churn_leaderboard.csv", "text/csv")

with tab_rev:
    try:
        m = drop_partial_last_month(load_revenue())
        fc = forecast_revenue(m, 3)
        fig = px.line(m, x="month", y="revenue", markers=True, height=420)
        fig.add_scatter(x=fc["month"], y=fc["revenue"], mode="lines+markers", name=f"forecast ({fc['method'].iloc[0]})",
                        line={"dash": "dash"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The final month is excluded when the data ends before the month is complete.")
    except Exception as exc:  # noqa: BLE001
        st.info(f"Revenue trend unavailable: {exc}")

with tab_explain:
    cid = st.selectbox("Customer ID", f.sort_values("at_risk_value", ascending=False)["customer_id"].head(500))
    row = df[df["customer_id"] == cid].iloc[0]
    st.write(f"**Churn probability:** {row['churn_probability']:.0%}  |  **Risk tier:** {row['risk_tier']}  |  "
             f"**Segment:** {row['segment_label']}")
    st.info(row["top_reasons"])
    contrib = pd.Series(json.loads(row["shap_json"])).sort_values(key=abs, ascending=False).head(10)[::-1]
    st.plotly_chart(px.bar(x=contrib.values, y=contrib.index, orientation="h", height=380,
                           labels={"x": "SHAP contribution (right = raises churn risk)", "y": ""}), use_container_width=True)
    if st.button("Also compute LIME explanation"):
        feats = json.loads(row["features_json"])
        rec = {k: feats[k] for k in ["recency_days", "frequency", "total_spend", "avg_spend", "historical_clv", "avg_basket_size",
                                     "freq_trend", "gap_variance", "seasonal_concentration", "return_rate",
                                     "discount_sensitivity", "avg_product_popularity"]}
        rec["country"] = row["country"]
        rec["category_affinity"] = {k[9:]: v for k, v in feats.items() if k.startswith("affinity_") and v > 0}
        r = api_post("/explain/customer", json={"record": rec, "method": "lime"})
        if r.ok:
            st.write(r.json()["explanation"])
            lime = pd.DataFrame(r.json()["contributions"]).head(10)
            st.dataframe(lime[["feature", "contribution"]], hide_index=True)
        else:
            st.error(r.text)
    meta = requests.get(f"{API_URL}/model/metadata", timeout=10).json() if os.environ.get("API_URL") else {}
    if meta.get("global_importance"):
        st.subheader("Global feature importance (mean |SHAP|)")
        gi = pd.DataFrame(meta["global_importance"]).head(10)[::-1]
        st.plotly_chart(px.bar(gi, x="mean_abs_shap", y="feature", orientation="h", height=360), use_container_width=True)

with tab_batch:
    st.caption("Upload a CSV of customer feature rows (same columns as the API). Rows with problems are listed, the rest are scored.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        r = api_post("/predict/batch", files={"file": (up.name, up.getvalue(), "text/csv")})
        if r.ok:
            body = r.json()
            st.success(f"Scored {body['n_scored']} of {body['n_rows']} rows.")
            if body["errors"]:
                st.warning("Some rows were rejected:")
                st.dataframe(pd.DataFrame(body["errors"]))
            res = pd.DataFrame(body["predictions"])
            res["recommended_categories"] = res["recommended_categories"].map(", ".join)
            st.dataframe(res, use_container_width=True)
            st.download_button("Download results (CSV)", res.to_csv(index=False), "batch_results.csv", "text/csv")
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Table

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4)
            top = res.sort_values("churn_probability", ascending=False).head(25)[
                ["customer_id", "churn_probability", "risk_tier", "predicted_clv_90d"]].round(3)
            doc.build([Paragraph("Batch scoring summary", getSampleStyleSheet()["Title"]),
                       Paragraph(f"Rows scored: {len(res)}; high risk: {(res['risk_tier'] == 'High').sum()}",
                                 getSampleStyleSheet()["Normal"]),
                       Table([list(top.columns)] + top.astype(str).values.tolist())])
            st.download_button("Download summary (PDF)", buf.getvalue(), "batch_summary.pdf", "application/pdf")
        else:
            detail = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
            st.error(f"Upload rejected: {detail}")
