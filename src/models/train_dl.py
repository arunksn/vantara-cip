"""Deep learning models (TensorFlow/Keras): feed-forward ANN, LSTM on purchase sequences, autoencoder.

Framework choice: TensorFlow (Keras API) is used consistently for all three models (PRD 21.4).
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline as SkPipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers, regularizers  # noqa: E402

from src.models.common import classification_metrics, select_threshold  # noqa: E402
from src.utils.config import artifacts_path, path_of  # noqa: E402
from src.utils.experiment_log import ExperimentLogger  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


# ----------------------------------------------------------------------------- builders
def build_ann(input_dim: int, cfg: dict[str, Any]) -> keras.Model:
    """Feed-forward network with batch normalisation, dropout and L1/L2 regularisation."""
    c = cfg["deep_learning"]["ann"]
    reg = regularizers.l1_l2(l1=c["l1"], l2=c["l2"])
    inp = keras.Input(shape=(input_dim,))
    x = inp
    for units in c["hidden_units"]:
        x = layers.Dense(units, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(c["dropout"])(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out, name="churn_ann")
    model.compile(optimizer=keras.optimizers.Adam(c["learning_rate"]), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    return model


def build_lstm(seq_len: int, n_categories: int, cfg: dict[str, Any]) -> keras.Model:
    """LSTM over [amount, gap, is_event] steps plus a category embedding."""
    c = cfg["deep_learning"]["lstm"]
    num_in = keras.Input(shape=(seq_len, 3), name="num")
    cat_in = keras.Input(shape=(seq_len,), dtype="int32", name="cat")
    emb = layers.Embedding(n_categories, c["embedding_dim"])(cat_in)
    x = layers.Concatenate()([num_in, emb])
    x = layers.LSTM(c["units"], dropout=c["dropout"])(x)
    x = layers.Dropout(c["dropout"])(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(c["l2"]))(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model([num_in, cat_in], out, name="churn_lstm")
    model.compile(optimizer=keras.optimizers.Adam(c["learning_rate"]), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    return model


def build_autoencoder(input_dim: int, cfg: dict[str, Any]) -> keras.Model:
    """Symmetric dense autoencoder trained to reconstruct scaled spending-pattern features."""
    c = cfg["deep_learning"]["autoencoder"]
    enc = c["encoder_units"]
    inp = keras.Input(shape=(input_dim,))
    x = inp
    for u in enc:
        x = layers.Dense(u, activation="relu")(x)
    for u in reversed(enc[:-1]):
        x = layers.Dense(u, activation="relu")(x)
    out = layers.Dense(input_dim)(x)
    model = keras.Model(inp, out, name="spend_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(c["learning_rate"]), loss="mse")
    return model


# ----------------------------------------------------------------------------- helpers
def _fit(model: keras.Model, x_tr: Any, y_tr: Any, x_va: Any, y_va: Any, c: dict[str, Any],
         class_weight: dict[int, float] | None, epochs: int | None = None) -> dict[str, list[float]]:
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=c["patience"], restore_best_weights=True)
    hist = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=epochs or c["epochs"],
                     batch_size=c["batch_size"], callbacks=[es], class_weight=class_weight, verbose=0)
    return {k: [float(v) for v in vals] for k, vals in hist.history.items()}


def _class_weights(y: np.ndarray) -> dict[int, float]:
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return {0: float(w[0]), 1: float(w[1])}


def plot_loss_curve(history: dict[str, list[float]], title: str, path: Any) -> None:
    """Save the training-vs-validation loss curve (convergence / overfitting check)."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(history["loss"], label="training loss")
    ax.plot(history["val_loss"], label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _subset(x: Any, idx: np.ndarray) -> Any:
    if isinstance(x, dict):
        return {k: v[idx] for k, v in x.items()}
    return x[idx]


def _cv_auc(builder: Any, x_tr: Any, y_tr: np.ndarray, c: dict[str, Any], cfg: dict[str, Any],
            use_smote: bool) -> tuple[float, float]:
    """5-fold stratified CV ROC-AUC on the training set."""
    skf = StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=cfg["project"]["seed"])
    cap = cfg["deep_learning"]["cv_epochs_cap"]
    aucs = []
    for tr, va in skf.split(np.zeros(len(y_tr)), y_tr):
        keras.utils.set_random_seed(cfg["project"]["seed"])
        model = builder()
        xt, yt = _subset(x_tr, tr), y_tr[tr]
        if use_smote:
            xt, yt = SMOTE(random_state=cfg["project"]["seed"]).fit_resample(xt, yt)
        _fit(model, xt, yt, _subset(x_tr, va), y_tr[va], c, None if use_smote else _class_weights(yt), epochs=cap)
        aucs.append(roc_auc_score(y_tr[va], model.predict(_subset(x_tr, va), verbose=0).ravel()))
        keras.backend.clear_session()
    return float(np.mean(aucs)), float(np.std(aucs))


def _evaluate(name: str, model: keras.Model, x_va: Any, y_va: np.ndarray, x_te: Any, y_te: np.ndarray,
              cfg: dict[str, Any], exp: ExperimentLogger, history: dict[str, list[float]], cv: tuple[float, float],
              train_time: float) -> dict[str, Any]:
    p_va = model.predict(x_va, verbose=0).ravel()
    thr = select_threshold(y_va, p_va, cfg["imbalance"]["min_recall_target"])
    p_te = model.predict(x_te, verbose=0).ravel()
    val_m, test_m = classification_metrics(y_va, p_va, thr), classification_metrics(y_te, p_te, thr)
    extra = {"cv_roc_auc_mean": cv[0], "cv_roc_auc_std": cv[1], "epochs_run": len(history["loss"])}
    exp.log("churn", name, "val", {**val_m, **extra}, {}, train_time)
    exp.log("churn", name, "test", {**test_m, **extra}, {}, train_time)
    logger.info("trained %s: cv_auc=%.3f val_auc=%.3f test_auc=%.3f test_recall=%.3f", name, cv[0],
                val_m["roc_auc"], test_m["roc_auc"], test_m["recall"])
    return {"name": name, "threshold": thr, "val": val_m, "test": test_m, "cv_auc_mean": cv[0], "cv_auc_std": cv[1],
            "p_val": p_va, "p_test": p_te, "history": history, "train_time": train_time}


# ----------------------------------------------------------------------------- public training API
def train_ann(splits_xy: dict[str, tuple[np.ndarray, np.ndarray]], cfg: dict[str, Any], exp: ExperimentLogger) -> dict[str, Any]:
    """Train the feed-forward ANN on the scaled engineered features."""
    c = cfg["deep_learning"]["ann"]
    seed = cfg["project"]["seed"]
    smote = cfg["imbalance"]["method"] == "smote"
    X_tr, y_tr = splits_xy["train"]
    scaler = StandardScaler().fit(X_tr)
    Xs = {k: scaler.transform(v[0]).astype("float32") for k, v in splits_xy.items()}
    ys = {k: v[1] for k, v in splits_xy.items()}
    joblib.dump(scaler, artifacts_path(cfg, "deep_learning", "ann_scaler.joblib"))

    def builder() -> keras.Model:
        return build_ann(Xs["train"].shape[1], cfg)

    cv = _cv_auc(builder, Xs["train"], ys["train"], c, cfg, smote)
    keras.utils.set_random_seed(seed)
    model = builder()
    xt, yt = Xs["train"], ys["train"]
    if smote:
        xt, yt = SMOTE(random_state=seed).fit_resample(xt, yt)
    t0 = time.time()
    hist = _fit(model, xt, yt, Xs["val"], ys["val"], c, None if smote else _class_weights(yt))
    tt = time.time() - t0
    plot_loss_curve(hist, "ANN: training vs validation loss", _fig(cfg, "loss_curve_ann.png"))
    model.save(artifacts_path(cfg, "deep_learning", "ann.keras"))
    return _evaluate("ann", model, Xs["val"], ys["val"], Xs["test"], ys["test"], cfg, exp, hist, cv, tt)


def train_lstm(seq: dict[str, dict[str, np.ndarray]], y: dict[str, np.ndarray], n_categories: int,
               cfg: dict[str, Any], exp: ExperimentLogger) -> dict[str, Any]:
    """Train the LSTM on time-ordered purchase sequences (amount, gap-in-days, category)."""
    c = cfg["deep_learning"]["lstm"]
    seed = cfg["project"]["seed"]
    seq_len = seq["train"]["num"].shape[1]

    def builder() -> keras.Model:
        return build_lstm(seq_len, n_categories, cfg)

    cv = _cv_auc(builder, seq["train"], y["train"], c, cfg, False)
    keras.utils.set_random_seed(seed)
    model = builder()
    t0 = time.time()
    hist = _fit(model, seq["train"], y["train"], seq["val"], y["val"], c, _class_weights(y["train"]))
    tt = time.time() - t0
    plot_loss_curve(hist, "LSTM: training vs validation loss", _fig(cfg, "loss_curve_lstm.png"))
    model.save(artifacts_path(cfg, "deep_learning", "lstm.keras"))
    return _evaluate("lstm", model, seq["val"], y["val"], seq["test"], y["test"], cfg, exp, hist, cv, tt)


def make_anomaly_preprocessor() -> SkPipeline:
    """log1p + standard scaling for the autoencoder's spending-pattern features."""
    return SkPipeline([("log", FunctionTransformer(np.log1p, validate=False)), ("scale", StandardScaler())])


def train_autoencoder(frames: dict[str, Any], cfg: dict[str, Any], exp: ExperimentLogger) -> dict[str, Any]:
    """Unsupervised anomaly detector; threshold validated on held-out customers.

    ``frames`` maps split name to a DataFrame containing the configured autoencoder features.
    """
    c = cfg["deep_learning"]["autoencoder"]
    cols = c["features"]
    seed = cfg["project"]["seed"]
    pre = make_anomaly_preprocessor().fit(frames["train"][cols].clip(lower=0))
    Z = {k: pre.transform(v[cols].clip(lower=0)).astype("float32") for k, v in frames.items()}
    keras.utils.set_random_seed(seed)
    model = build_autoencoder(len(cols), cfg)
    t0 = time.time()
    hist = _fit(model, Z["train"], Z["train"], Z["val"], Z["val"], c, None)
    tt = time.time() - t0

    resid_tr = model.predict(Z["train"], verbose=0) - Z["train"]
    scale = np.maximum(resid_tr.var(axis=0), 1e-6)   # per-feature noise level: stops noisy features dominating the score

    def err(z: np.ndarray) -> np.ndarray:
        return np.mean(((model.predict(z, verbose=0) - z) ** 2) / scale, axis=1)

    e_tr, e_va, e_te = err(Z["train"]), err(Z["val"]), err(Z["test"])
    thr = float(np.percentile(e_tr, c["anomaly_percentile"]))
    flagged_te = e_te > thr
    # Independent validation of the flags: rank agreement with Mahalanobis distance and an Isolation Forest.
    from scipy.stats import spearmanr
    from sklearn.ensemble import IsolationForest

    zt = Z["test"]
    cov = np.cov(Z["train"].T) + 1e-6 * np.eye(zt.shape[1])
    cen = zt - Z["train"].mean(0)
    maha = np.einsum("ij,jk,ik->i", cen, np.linalg.inv(cov), cen)
    iso = -IsolationForest(random_state=seed).fit(Z["train"]).score_samples(zt)
    top_iso = np.argsort(iso)[::-1][: max(int(0.05 * len(iso)), 1)]
    report = {
        "threshold": thr, "percentile": c["anomaly_percentile"],
        "flag_rate_train": float((e_tr > thr).mean()), "flag_rate_val": float((e_va > thr).mean()),
        "flag_rate_test": float(flagged_te.mean()), "n_flagged_test": int(flagged_te.sum()),
        "spearman_vs_mahalanobis": float(spearmanr(e_te, maha).correlation),
        "spearman_vs_isolation_forest": float(spearmanr(e_te, iso).correlation),
        "flagged_in_isolation_forest_top5pct": float(np.isin(np.flatnonzero(flagged_te), top_iso).mean()) if flagged_te.any() else None,
        "final_val_loss": hist["val_loss"][-1], "expected_flag_rate": 1 - c["anomaly_percentile"] / 100.0,
        "score": "mean over features of squared reconstruction error / training residual variance",
    }
    exp.log("anomaly", "autoencoder", "test", report, {"features": cols}, tt)
    plot_loss_curve(hist, "Autoencoder: training vs validation loss", _fig(cfg, "loss_curve_autoencoder.png"))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(np.log10(e_tr + 1e-9), bins=60, color="#4c72b0")
    ax.axvline(np.log10(thr + 1e-9), color="crimson", ls="--", label=f"threshold (p{c['anomaly_percentile']:.0f})")
    ax.set_xlabel("log10 scaled reconstruction error")
    ax.set_ylabel("customers")
    ax.set_title("Autoencoder reconstruction error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_fig(cfg, "autoencoder_error.png"), dpi=130)
    plt.close(fig)
    model.save(artifacts_path(cfg, "deep_learning", "autoencoder.keras"))
    joblib.dump({"preprocessor": pre, "threshold": thr, "features": cols, "resid_scale": scale}, artifacts_path(cfg, "deep_learning", "anomaly.joblib"))
    with open(artifacts_path(cfg, "metrics", "anomaly_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("autoencoder: threshold=%.5f flag_rate_val=%.4f flag_rate_test=%.4f", thr, report["flag_rate_val"],
                report["flag_rate_test"])
    return {"model": model, "report": report, "history": hist}


def _fig(cfg: dict[str, Any], name: str) -> Any:
    p = path_of(cfg, "figures_dir") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
