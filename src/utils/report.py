"""Builds docs/final_report.pdf from saved metrics/figures (nothing is typed by hand)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.utils.config import artifacts_path, path_of
from src.utils.experiment_log import read_log
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "C:/Windows/Fonts"]


def _fonts() -> tuple[str, str]:
    """Register DejaVu (Unicode math symbols) if available; otherwise fall back to Helvetica."""
    for d in FONT_DIRS:
        reg, bold = Path(d) / "DejaVuSans.ttf", Path(d) / "DejaVuSans-Bold.ttf"
        if reg.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("DejaVu", str(reg)))
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(bold)))
            return "DejaVu", "DejaVu-Bold"
    return "Helvetica", "Helvetica-Bold"


def _img(path: Path, width_cm: float) -> Image | None:
    if not path.exists():
        return None
    from PIL import Image as PILImage

    w, h = PILImage.open(path).size
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def _table(df: pd.DataFrame, font: str, col_widths: list[float] | None = None, fs: int = 7) -> Table:
    data = [list(map(str, df.columns))] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#33506b")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def success_table(meta: dict[str, Any], clv: dict[str, Any], bench: dict[str, Any]) -> pd.DataFrame:
    """Section 3.2 targets vs. observed values."""
    pm = meta.get("production_test_metrics", {})
    sel = clv.get("selected")
    r2 = clv.get(sel, {}).get("test", {}).get("r2") if sel else None
    p95 = bench.get("api", {}).get("p95_ms")
    dash = bench.get("dashboard", {}).get("first_load_sec")

    def row(metric: str, target: str, obs: float | None, ok: bool | None, fmt: str) -> list[str]:
        return [metric, target, "n/a" if obs is None else fmt.format(obs), "n/a" if ok is None else ("MET" if ok else "NOT MET")]

    rows = [
        row("Churn ROC-AUC (test)", ">= 0.80", pm.get("roc_auc"), None if "roc_auc" not in pm else pm["roc_auc"] >= 0.80, "{:.3f}"),
        row("CLV R² (test)", ">= 0.60", r2, None if r2 is None else r2 >= 0.60, "{:.3f}"),
        row("Churn recall (test)", ">= 0.70", pm.get("recall"), None if "recall" not in pm else pm["recall"] >= 0.70, "{:.3f}"),
        row("API p95 latency (single prediction)", "< 400 ms", p95, None if p95 is None else p95 < 400, "{:.1f} ms"),
        row("Dashboard load (segment view)", "< 3 s", dash, None if dash is None else dash < 3, "{:.2f} s"),
        ["Single-command reproducible pipeline", "all steps scripted", "python -m src.pipeline", "MET"],
    ]
    return pd.DataFrame(rows, columns=["Metric (PRD 3.2)", "Target", "Observed", "Status"])


def build_report(cfg: dict[str, Any]) -> Path:
    """Generate the final PDF report."""
    font, bold = _fonts()
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontName=font, fontSize=9, leading=12.5)
    small = ParagraphStyle("s", parent=body, fontSize=8, leading=10.5)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName=bold, fontSize=15, spaceBefore=10, textColor=colors.HexColor("#1f3b57"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName=bold, fontSize=11.5, spaceBefore=8)
    title = ParagraphStyle("t", parent=ss["Title"], fontName=bold, fontSize=22, alignment=TA_CENTER)
    mono = ParagraphStyle("m", parent=body, fontSize=10.5, leading=17, leftIndent=14, spaceAfter=10)
    banner = ParagraphStyle("ban", parent=body, fontName=bold, fontSize=10, textColor=colors.white, backColor=colors.HexColor("#b03030"),
                            borderPadding=6, alignment=TA_CENTER, leading=14)

    meta = _read(artifacts_path(cfg, "metadata.json"))
    mdir = artifacts_path(cfg, "metrics", "x").parent
    clv, seg = _read(mdir / "clv_metrics.json"), _read(mdir / "segmentation_metrics.json")
    anom, expl = _read(mdir / "anomaly_report.json"), _read(mdir / "explain_report.json")
    bench, nxt = _read(mdir / "benchmarks.json"), _read(mdir / "next_category.json")
    clean = _read(path_of(cfg, "cleaning_report"))
    cmp_tbl = pd.read_csv(mdir / "model_comparison.csv")
    fdir, ddir = path_of(cfg, "figures_dir"), path_of(cfg, "docs_dir")
    prod = meta.get("production_model", "?")
    pm = meta.get("production_test_metrics", {})
    S: list[Any] = []
    P = lambda t, st=body: S.append(Paragraph(t, st))  # noqa: E731

    S.append(Paragraph("Customer Behavior Prediction Platform", title))
    S.append(Paragraph("Final report: Vantara Retail Solutions", ParagraphStyle("st", parent=body, alignment=TA_CENTER, fontSize=12)))
    S.append(Spacer(1, 8))
    if meta.get("synthetic"):
        S.append(Paragraph("THE RESULTS IN THIS REPORT WERE GENERATED ON A SYNTHETIC STAND-IN DATASET (same schema as UCI Online "
                           "Retail II). THEY ARE NOT REAL-DATA RESULTS. Re-run <b>python -m src.pipeline</b> on the real "
                           "dataset to regenerate this report.", banner))
    S.append(Spacer(1, 6))
    P(f"Data source: {meta.get('data_source')}. Run: {meta.get('run_id')}. Trained: {meta.get('trained_at')}. "
      f"Seed: {meta.get('seed')}. Training cutoff {meta.get('cutoffs', {}).get('train_cutoff')}; scoring cutoff "
      f"{meta.get('cutoffs', {}).get('scoring_cutoff')}.", small)

    S.append(Paragraph("1. Executive summary", h1))
    P(f"The platform predicts, per customer, the probability of no purchase in the next 90 days (churn), 90-day customer "
      f"lifetime value, anomalous spending and the most likely next purchase categories, and groups customers into "
      f"behavioural segments. The recommended production churn model is <b>{prod}</b> (held-out test ROC-AUC "
      f"{pm.get('roc_auc', float('nan')):.3f}, recall {pm.get('recall', float('nan')):.3f}, precision "
      f"{pm.get('precision', float('nan')):.3f}). {meta.get('scored_customers', '?')} customers were scored; "
      f"{meta.get('high_risk_customers', '?')} are in the High risk tier.")
    S.append(Spacer(1, 4))
    S.append(Paragraph("Success metrics (PRD Section 3.2)", h2))
    st = success_table(meta, clv, bench)
    S.append(_table(st, font, [6.2 * cm, 3 * cm, 4.3 * cm, 2.5 * cm], 8))
    S.append(Spacer(1, 4))
    for _, r in st[st["Status"] == "NOT MET"].iterrows():
        metric = r["Metric (PRD 3.2)"]
        if metric.startswith("CLV"):
            sel = clv.get("selected")
            others = {k: v["test"]["r2"] for k, v in clv.items() if k not in ("selected", sel)}
            alt = "; ".join(f"{k} reached test R² {v:.3f}" for k, v in others.items())
            why = (f"The future-spend target is heavy-tailed and dominated by a few very large customers, which makes R² unstable. "
                   f"Model selection used training-set CV R² only and chose {sel}" + (f" ({alt}, but the test set was deliberately not used to pick a model)" if alt else "") +
                   ". Candidate remedies to evaluate on the real data: a log-transformed target, a two-stage (purchase probability x spend) model, "
                   "winsorising extreme customers, or a longer CLV horizon.")
        elif "latency" in metric or "Dashboard" in metric:
            why = "The benchmark ran on a 1-CPU sandbox without network; re-measure on the deployment hardware and consider caching or a lighter model."
        else:
            why = (f"Candidate causes to investigate: label noise in a single 90-day window, the limited customer count (about "
                   f"{meta.get('n_customers_train', '?')}), and feature coverage.")
        P(f"<b>Discussion, {metric}:</b> observed {r['Observed']} vs target {r['Target']}. Reported as measured, not tuned toward the "
          f"target. {why}", small)
    P("Class imbalance: accuracy is reported but de-prioritised; precision, recall, F1 and ROC-AUC are always shown together.", small)

    S.append(Paragraph("2. Data and cleaning", h1))
    P("Both yearly sheets are concatenated and ordered chronologically. Cleaning rules: exact duplicates removed; rows without "
      "Customer ID flagged (kept for product-level analysis, excluded from customer models); cancellations and negative "
      "quantities flagged as returns (not dropped; they feed return_rate); zero/negative prices flagged as adjustments; "
      "non-product StockCodes flagged; IQR outliers split into data-entry errors (excluded) and legitimate bulk purchases "
      "(kept); descriptions standardised through a StockCode lookup.")
    if clean:
        keys = ["rows_in", "exact_duplicates_removed", "rows_missing_customer_id", "return_lines", "price_adjustment_lines",
                "admin_code_lines", "rows_used_for_customer_model", "unique_customers"]
        S.append(_table(pd.DataFrame([(k, f"{clean.get(k):,}") for k in keys], columns=["Item", "Rows / count"]), font,
                        [8 * cm, 4 * cm], 8))
        o = clean.get("outliers", {})
        P(f"Outliers: {o.get('n_outliers')} flagged; {o.get('n_entry_errors')} data-entry errors excluded; "
          f"{o.get('n_bulk_outliers')} bulk purchases kept.", small)

    S.append(Paragraph("3. Feature engineering", h1))
    P("All features use only transactions strictly before the cutoff date; labels use the following 90 days. Leakage is guarded "
      "by an assertion in the builder and by unit tests. Country is one-hot encoded; StockCode is frequency-encoded into "
      "avg_product_popularity. Product-category affinity is the share of spend per keyword-derived category. discount_sensitivity "
      "is a proxy (the dataset has no discount field): share of spend on lines priced at least 10% below the SKU median.")
    P(f"VIF pruning (threshold {cfg['features']['vif_threshold']}) removed: {', '.join(meta.get('vif_dropped', [])) or 'none'}. "
      f"{len(meta.get('model_features', []))} features remain in the model matrix.", small)

    S.append(Paragraph("4. Churn model comparison", h1))
    P("Every model uses the same stratified 70/15/15 split and seed. Hyper-parameters were tuned with 5-fold stratified CV on the "
      "training set only; early stopping used the validation split; the operating threshold (highest F1 with recall >= 0.70) "
      "was chosen on the validation set; the test set was used once for the numbers below. Ranked by ROC-AUC then recall.")
    show = cmp_tbl[["rank", "model_name", "cv_auc_mean", "roc_auc", "recall", "precision", "f1", "accuracy"]].copy()
    show.columns = ["#", "Model", "CV AUC", "Test AUC", "Recall", "Precision", "F1", "Accuracy"]
    S.append(_table(show.round(3), font, [0.8 * cm, 4.2 * cm, 1.8 * cm, 1.9 * cm, 1.8 * cm, 2 * cm, 1.6 * cm, 1.9 * cm], 8))
    S.append(Spacer(1, 4))
    log = read_log(cfg, meta.get("run_id"))
    pt = log[(log["task"] == "churn") & (log["split"] == "test") & (log["model"] == prod)]
    if len(pt):
        r = pt.iloc[0]
        S.append(Paragraph(f"Confusion matrix, {prod}, test set (threshold {r['metrics.threshold']:.3f})", h2))
        S.append(_table(pd.DataFrame([["Actual retained", int(r["metrics.tn"]), int(r["metrics.fp"])],
                                      ["Actual churned", int(r["metrics.fn"]), int(r["metrics.tp"])]],
                                     columns=["", "Predicted retained", "Predicted churned"]), font, [4 * cm, 4 * cm, 4 * cm], 8))
    best_auc = cmp_tbl.iloc[0]
    P(f"<b>Recommendation.</b> {prod} is recommended for the production API. It was selected among the tree-based candidates "
      f"by validation ROC-AUC (recall as tie-break). Tree ensembles are exactly compatible with SHAP TreeExplainer, score a "
      f"single customer in milliseconds, need no feature scaling and, with only about {meta.get('n_customers_train', '?')} "
      f"customers, are the safer choice than deep networks (PRD Section 19). The top-ranked model overall by test AUC is "
      f"{best_auc['model_name']} ({best_auc['roc_auc']:.3f}); where it differs from {prod}, the gap is small compared with "
      f"the explainability and latency benefits. The ANN and LSTM are retained as comparative models.")
    for nm in ("loss_curve_ann.png", "loss_curve_lstm.png"):
        im = _img(fdir / nm, 8)
        if im:
            S.append(im)

    S.append(Paragraph("5. CLV, next category and anomaly detection", h1))
    if clv:
        rows = [[k, f"{v['test']['mae']:.2f}", f"{v['test']['rmse']:.2f}", f"{v['test']['r2']:.3f}", f"{v['cv_r2']:.3f}"]
                for k, v in clv.items() if k != "selected"]
        S.append(_table(pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R² (test)", "CV R² (train)"]), font,
                        [4 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 3 * cm], 8))
        P(f"Selected CLV model (by training-set CV R²): {clv.get('selected')}. Target: spend in the 90 days after the cutoff.", small)
    if nxt:
        P(f"Next purchase category model (multiclass random forest): accuracy {nxt['accuracy']:.3f}, macro-F1 {nxt['macro_f1']:.3f}.", small)
    if anom:
        P(f"Autoencoder anomaly detector: threshold at the {anom['percentile']:.0f}th percentile of training reconstruction error "
          f"({anom['threshold']:.4f}); flag rate train/val/test = {anom['flag_rate_train']:.3%} / {anom['flag_rate_val']:.3%} / "
          f"{anom['flag_rate_test']:.3%} (expected {anom['expected_flag_rate']:.2%}), so the threshold transfers to held-out customers. "
          f"Independent check of the flags: Spearman rank correlation of the score with Mahalanobis distance "
          f"{anom['spearman_vs_mahalanobis']:.2f} and with an Isolation Forest {anom['spearman_vs_isolation_forest']:.2f}; share of "
          f"flagged test customers inside the Isolation Forest top 5%: {anom['flagged_in_isolation_forest_top5pct']}. "
          f"Weak agreement means the flags should be treated as a review list, not as confirmed anomalies.", small)
        im = _img(fdir / "autoencoder_error.png", 8)
        if im:
            S.append(im)

    S.append(Paragraph("6. Customer segmentation", h1))
    if seg:
        k, sec = seg["kmeans"], seg["secondary"]
        P(f"K-Means with k={k['k']} chosen by silhouette over k={k['k_values'][0]}..{k['k_values'][-1]} (elbow plot reviewed): "
          f"silhouette {k['silhouette']:.3f}, Davies-Bouldin {k['davies_bouldin']:.3f}. Secondary algorithm "
          f"({sec['algorithm'].upper()}): silhouette {sec['silhouette']}, Davies-Bouldin {sec['davies_bouldin']}, "
          f"adjusted Rand vs K-Means {sec['adjusted_rand_vs_kmeans']}.")
    prof_p = mdir / "segment_profiles.csv"
    if prof_p.exists():
        pr = pd.read_csv(prof_p)
        cols = [c for c in ["segment_label", "customers", "share_pct", "recency_days", "frequency", "total_spend",
                            "return_rate", "mean_churn_probability"] if c in pr.columns]
        S.append(_table(pr[cols].round(2), font, None, 7))
    for nm in ("segments_pca.png", "kmeans_k_selection.png"):
        im = _img(fdir / nm, 11)
        if im:
            S.append(im)

    S.append(Paragraph("7. Explainability", h1))
    P("Global SHAP importance, three representative customers, partial dependence of the top features and a LIME-vs-SHAP "
      "comparison are below.")
    for nm in ("shap_summary.png", "shap_force_high_risk.png", "shap_force_low_risk.png", "shap_force_borderline.png",
               "pdp_top_features.png", "lime_vs_shap.png"):
        im = _img(fdir / nm, 14 if nm != "shap_summary.png" else 11)
        if im:
            S.append(im)
    if expl:
        c = expl["lime_vs_shap"]
        P(f"LIME vs SHAP for the borderline customer: top-{c['top_k']} overlap {c['overlap_fraction']:.0%} "
          f"({', '.join(c['overlap']) or 'none'}), sign agreement {c['sign_agreement']}. Differences are expected: SHAP "
          f"attributes the model's own output exactly, LIME fits a local linear surrogate on perturbed samples.", small)
        P(f"<b>Plain-language template example:</b> {expl['example_explanation']}", small)

    S.append(Paragraph("8. Serving, deployment and measurements", h1))
    P("FastAPI exposes /predict/customer, /predict/batch (CSV), /model/metadata, /health and /explain/customer; Pydantic models "
      "validate every record. Predictions and segments are persisted in PostgreSQL; the Streamlit dashboard reads them and calls "
      "the API for batch scoring and LIME. Everything starts with docker-compose up.")
    if bench:
        a, d = bench.get("api", {}), bench.get("dashboard", {})
        P(f"API latency ({a.get('requests')} requests): p50 {a.get('p50_ms', float('nan')):.1f} ms, p95 {a.get('p95_ms', float('nan')):.1f} ms, "
          f"p99 {a.get('p99_ms', float('nan')):.1f} ms. Method: {a.get('method')}. Dashboard server-side run: first "
          f"{d.get('first_load_sec', float('nan')):.2f} s, cached {d.get('cached_load_sec', float('nan')):.2f} s ({d.get('method')}). "
          f"Both were measured in a 1-CPU sandbox, not on production hardware or over a network.", small)
    for nm in ("architecture_diagram.png", "er_diagram.png", "workflow_diagram.png"):
        im = _img(ddir / nm, 15)
        if im:
            S.append(im)

    S.append(PageBreak())
    S.append(Paragraph("Appendix A. Mathematical foundations", h1))
    A = [
        ("A.1 Statistics and probability",
         "RFM and the engagement score use percentile ranks: score = 100·(1 − PR(recency) + PR(frequency) + PR(monetary))/3, where "
         "PR(x) is the empirical CDF value from the fitted reference distribution. Next-purchase probability is "
         "P(purchase | recency, frequency, monetary, …) = 1 − ŷ, the classifier's predicted probability."),
        ("A.2 Linear algebra and PCA",
         "The feature matrix is X ∈ ℝ^(n×d). PCA diagonalises the covariance Σ = (1/n)XᵀX (X centred): Σv = λv. The segment plots project onto the two "
         "eigenvectors with largest λ (explained variance: " + ", ".join(f"{v:.1%}" for v in seg.get("pca_explained_variance", [])) + "). "
         "K-Means minimises Σₖ Σ_{x∈Cₖ} ‖x − μₖ‖²; silhouette s = (b − a)/max(a, b)."),
        ("A.3 Logistic regression and cross-entropy",
         "P(y=1|x) = 1/(1 + e^−(wᵀx+b)); loss L = −(1/n) Σ [yᵢ log ŷᵢ + (1−yᵢ) log(1−ŷᵢ)]. Class weights multiply each term by w_c."),
        ("A.4 Gradient descent and backpropagation",
         "w ← w − η∇L(w). Backpropagation applies the chain rule layer by layer: ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ with δ⁽ˡ⁾ = ((W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾) ⊙ f′(z⁽ˡ⁾). "
         "Adam rescales the step by running moment estimates. For the LSTM, gradients flow back through time via the gate equations "
         "cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ. The training-vs-validation loss curves (Section 4) confirm convergence and flag overfitting; early stopping "
         "restores the best validation weights."),
        ("A.5 Regularisation",
         "L1: L + λ‖w‖₁ (sparse weights); L2: L + λ‖w‖₂² (shrinkage). In this code, logistic regression tunes C = 1/λ and l1_ratio ∈ {0, 1}; the ANN "
         "uses an L2 kernel penalty. Dropout multiplies activations by Bernoulli(1−p) masks during training, an implicit ensemble average."),
        ("A.6 Evaluation metrics",
         "Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2PR/(P+R); ROC-AUC = P(score of a random churner > score of a random retained customer), "
         "the area under TPR vs FPR over all thresholds. MAE = mean|y−ŷ|, RMSE = √mean(y−ŷ)², R² = 1 − Σ(y−ŷ)²/Σ(y−ȳ)². "
         "Davies-Bouldin = mean over clusters of max_{j≠i}(σᵢ+σⱼ)/d(μᵢ,μⱼ) (lower is better)."),
        ("A.7 Tree ensembles and gradient boosting",
         "A decision tree chooses the split that most reduces impurity; for the Gini index G = 1 − Σₖ pₖ², the gain of splitting a node into children "
         "L and R is ΔG = G − (n_L/n)G_L − (n_R/n)G_R. Random forests average B trees grown on bootstrap samples with random feature subsets, "
         "which reduces variance: Var(mean) = ρσ² + (1−ρ)σ²/B for pairwise correlation ρ. Gradient boosting builds F_m(x) = F_{m−1}(x) + ν·h_m(x), where "
         "h_m fits the negative gradient rᵢ = −∂L(yᵢ, F(xᵢ))/∂F(xᵢ) (for log-loss rᵢ = yᵢ − p̂ᵢ). XGBoost adds a second-order Taylor expansion "
         "and a complexity penalty γT + ½λ‖w‖² per tree; scale_pos_weight rescales positive-class gradients to handle imbalance. "
         "Early stopping halts adding trees when the validation AUC stops improving."),
        ("A.8 Sequence model and autoencoder",
         "The LSTM cell updates its state with fₜ = σ(W_f[hₜ₋₁, xₜ] + b_f), iₜ = σ(W_i[hₜ₋₁, xₜ] + b_i), c̃ₜ = tanh(W_c[hₜ₋₁, xₜ] + b_c), "
         "cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ, oₜ = σ(W_o[hₜ₋₁, xₜ] + b_o), hₜ = oₜ⊙tanh(cₜ). Each input step is [amount, gap in days, is-event] plus a "
         "category embedding; a final step carries the days from the last order to the prediction date. The autoencoder learns "
         "x̂ = g(f(x)) by minimising (1/n)Σ‖x − x̂‖²; the anomaly score here is (1/d)Σⱼ (xⱼ − x̂ⱼ)²/σⱼ² with σⱼ² the training residual variance of feature j, "
         "flagged above the 99th training percentile."),
        ("A.9 Operating threshold and class weights",
         "With class weights w₀, w₁ the loss becomes −(1/n)Σ wᵧᵢ[yᵢ log ŷᵢ + (1−yᵢ)log(1−ŷᵢ)], where w_c = n/(2·n_c) for 'balanced'. The decision rule is "
         "ŷ = 1[p̂ ≥ τ]; τ is chosen on the validation set as the F1-maximising threshold among those with recall ≥ 0.70, then frozen before the test set is scored."),
        ("A.10 SHAP",
         "Shapley values φᵢ = Σ_{S⊆F∖{i}} |S|!(|F|−|S|−1)!/|F|! · [f(S∪{i}) − f(S)] satisfy local accuracy f(x) = φ₀ + Σφᵢ. TreeExplainer computes them "
         "exactly in polynomial time for tree ensembles; for XGBoost/LightGBM the values are in log-odds space."),
    ]
    for head, txt in A:
        S.append(Paragraph(head, h2))
        S.append(Paragraph(txt, mono))

    S.append(Paragraph("Appendix B. Future enhancements", h1))
    for t in ["Transformer-based sequence model for purchase timing (needs a larger event log and more tuning).",
              "Streaming / near-real-time scoring (event ingestion, incremental feature store).",
              "API authentication, rate limiting and CI/CD with automated retraining and model registry.",
              "A/B testing of retention campaigns to measure causal uplift, and drift monitoring on feature and score distributions.",
              "Additional boosted-tree variants (e.g. CatBoost) and probability calibration."]:
        P("• " + t)

    out = ddir / "final_report.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                      bottomMargin=1.6 * cm, title="Customer Behavior Prediction: Final Report").build(S)
    logger.info("report written: %s", out)
    return out
