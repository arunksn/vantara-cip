# Vantara Customer Intelligence Platform

Predicts, per customer, **churn** (no purchase in the next 90 days), **90-day customer lifetime value**, **anomalous spending**
and **next-purchase categories**; segments customers; explains every prediction (SHAP, LIME, plain language); serves results
through a **FastAPI** service, **PostgreSQL** and a **Streamlit** dashboard. 

> **READ THIS FIRST: the results shipped in this repository come from a SYNTHETIC STAND-IN dataset.**
> The UCI server was unreachable from the build sandbox, so the pipeline was verified end to end on generated data with the
> same schema and the same data-quality problems as Online Retail II (`tests/synthetic.py`). All numbers in
> `docs/final_report.pdf`, `models_artifacts/` and `data/processed/` are labelled `SYNTHETIC` and are **not real-data results**.
> Run `python -m src.pipeline` with network access (or the xlsx, see below) to regenerate everything on the real dataset.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.pipeline            # whole pipeline, real data (Option B download, falls back to Option A xlsx)
python -m pytest --cov=src --cov=api            # tests (about 70% coverage gate in the PRD)
```

### Data (PRD Section 5)
* **Option B (default):** `ucimlrepo` fetches UCI dataset 502 and caches it in `data/raw/`.
* **Option A (fallback):** put `online_retail_II.xlsx` in `data/raw/` (both sheets are read and concatenated).
* `--input FILE` forces a specific CSV/XLSX; `--synthetic` labels the outputs as synthetic.
* Re-run single stages with `--stages data features train explain segment score report`.

### Run the stack
```bash
cp .env.example .env              # set POSTGRES_PASSWORD (credentials are never committed)
docker-compose up --build         # db -> init (loads PostgreSQL) -> api (:8000) -> dashboard (:8501)
```
* Dashboard: http://localhost:8501 · API docs (OpenAPI/Swagger): http://localhost:8000/docs
* Without Docker: `uvicorn api.main:app` and `DATABASE_URL=... streamlit run frontend/dashboard.py`
  (load the database first with `python -m src.models.batch_score --load-db`).

## Repository layout
```
config/config.yaml        every path, hyper-parameter grid and threshold (nothing hard-coded in scripts)
data/{raw,interim,processed}
notebooks/                01_eda, 02_feature_engineering, 03_model_experiments (read pipeline outputs only)
src/data                  load, clean, validate
src/features              builders (point-in-time), transform (encoders, engagement, VIF), sequences (LSTM)
src/models                train_classical, train_dl, common, predictor, batch_score
src/segmentation          K-Means + GMM/DBSCAN, profiling, labelling
src/explainability        SHAP, LIME, PDP, plain-language template
src/utils                 config, logging, experiment log, db, forecast, diagrams, report, benchmark
src/pipeline.py           single-command orchestrator
api/                      FastAPI app, routers, Pydantic schemas
frontend/dashboard.py     Streamlit app
models_artifacts/         fitted models, transformer, metrics/, experiment_log.jsonl, metadata.json
docs/                     final_report.pdf, architecture / ER / workflow diagrams, figures/
tests/                    test_features.py (incl. leakage), test_api.py, test_pipeline.py
```

## API
| Method & path | Purpose |
|---|---|
| `POST /predict/customer` | churn probability, risk tier (Low/Medium/High), next-purchase probability, 90-day CLV, anomaly flag, top-3 categories. `?persist=true` + `customer_id` stores it in PostgreSQL |
| `POST /predict/batch` | CSV upload; invalid rows are listed with row number and reasons, valid rows are scored |
| `POST /explain/customer` | SHAP or LIME contributions + plain-language explanation |
| `GET /model/metadata` | model name/version, training date, feature list, operating threshold, test metrics, global importance |
| `GET /health` | model loaded, database reachable |

```bash
curl -s localhost:8000/predict/customer -H 'content-type: application/json' -d '{
  "country":"United Kingdom","recency_days":45,"frequency":6,"total_spend":1450.5,"avg_spend":241.75,
  "historical_clv":1380,"avg_basket_size":120,"freq_trend":-0.1,"gap_variance":900,"seasonal_concentration":0.2,
  "return_rate":0.02,"discount_sensitivity":0.1,"avg_product_popularity":0.05,"category_affinity":{"lighting":0.3}}'
```
Malformed records return HTTP 422 with `{"detail": ..., "errors": [{"field": ..., "message": ...}]}`. Batch CSV columns are the
same names; category affinity is given as `affinity_<category>` columns.

## Methodology and interpretation choices (where the PRD left room)
* **Two snapshots.** Training snapshot: features before `cutoff = last date - 90 days`, labels from the 90 days after. Scoring
  snapshot: features as of the end of the data (stored in PostgreSQL, shown in the dashboard).
* **Leakage control.** Features see only rows strictly before the cutoff (asserted in code; tests alter future rows and require identical features).
* **CLV target** = net spend in the 90 days after the cutoff (a *future-value* definition). `historical_clv` (a feature) = net revenue to date.
* **Product categories** (the dataset has none) come from keyword rules in `config.yaml`; **discount sensitivity** is a proxy
  (share of spend on lines at least 10% below the SKU median price). **Engagement score** = mean percentile rank of recency (inverted), frequency, spend.
* **Imbalance:** class weights on training data only (`imbalance.method: smote` switches to SMOTE inside each training fold).
* **Operating threshold** per model = highest-F1 threshold on the *validation* set among those with recall >= 0.70; the test set is used once.
* **Production model** = best validation ROC-AUC among tree models (SHAP-compatible); the final comparison table covers all 8 churn models.
* **VIF pruning** (threshold 10) drops redundant features from the model matrix; they are still computed and stored.
* **Risk tiers:** High >= operating threshold, Medium >= 60% of it, else Low.
* **Anomaly score** = autoencoder reconstruction error scaled per feature by its training noise; threshold = 99th percentile of training scores.
* **Deep-learning framework:** TensorFlow/Keras throughout. Seeds are fixed in `config.yaml`; exact bit-for-bit reproducibility across machines is not guaranteed for TensorFlow.

## Deviations from the PRD folder structure / scope (each is deliberate)
* Added `src/pipeline.py` and `src/models/{common,predictor,batch_score}.py`, `src/utils/*`, `src/features/{transform,sequences}.py`,
  `src/segmentation/cluster.py`, `tests/{conftest,synthetic,test_pipeline}.py`, `pyproject.toml`, `.env.example`, `.dockerignore`: needed for the single-command pipeline, shared train/serve logic and the required tests.
* Added a one-shot `init` service to `docker-compose.yml` (loads PostgreSQL) and a `POST /explain/customer` endpoint (the dashboard needs on-demand LIME).
* Figures are in `docs/figures/`; metrics JSON/CSV in `models_artifacts/metrics/`.
* **Not delivered:** the recorded walkthrough video (cannot be produced in the build environment).

## Known limitations
* Metrics in this repository are from synthetic data (see top). Targets in PRD 3.2 must be re-checked on the real dataset; `docs/final_report.pdf` regenerates the pass/fail table automatically.
* The autoencoder's anomaly flags agree only weakly with independent outlier detectors on the synthetic data; treat flags as a review list.
* CLV R-squared is sensitive to a few very large customers (heavy-tailed target).
* API latency and dashboard load time are measured in-process (no network, 1 CPU); re-measure on your deployment hardware.
* `docker-compose` and PostgreSQL were **not** executed in the build environment (no Docker); the database layer was tested against SQLite via SQLAlchemy. Please run `docker-compose up --build` once and report any issue.
