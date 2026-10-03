"""Customer segmentation: K-Means (primary), Gaussian Mixture or DBSCAN (secondary), optional hierarchical check.

k is chosen from the elbow curve and silhouette score (not fixed arbitrarily): the k with the highest silhouette
within [k_min, k_max]; the elbow curve is plotted alongside for review.
"""
from __future__ import annotations

from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from sklearn.cluster import DBSCAN, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import davies_bouldin_score, silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.utils.config import artifacts_path, path_of  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def _prep(df: pd.DataFrame, cfg: dict[str, Any], scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    scfg = cfg["segmentation"]
    Z = df[scfg["features"]].copy().astype(float)
    for col in scfg["log_features"]:
        Z[col] = np.log1p(Z[col].clip(lower=0))
    if scaler is None:
        scaler = StandardScaler().fit(Z)
    return scaler.transform(Z), scaler


def _quality(Z: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    mask = labels >= 0
    if len(set(labels[mask])) < 2:
        return {"silhouette": None, "davies_bouldin": None}
    return {"silhouette": float(silhouette_score(Z[mask], labels[mask])),
            "davies_bouldin": float(davies_bouldin_score(Z[mask], labels[mask]))}


def label_segments(df: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Business-readable names from cluster means relative to the whole base (z-scores of cluster means)."""
    d = df.assign(_seg=labels)
    means = d.groupby("_seg")[["recency_days", "frequency", "total_spend", "return_rate", "seasonal_concentration"]].mean()
    z = (means - means.mean()) / means.std(ddof=0).replace(0, 1)
    names: dict[int, str] = {}
    for seg, r in z.iterrows():
        if r["total_spend"] > 0.5 and r["recency_days"] > 0.3:
            n = "Lapsed High-Value"
        elif r["total_spend"] > 0.5:
            n = "Loyal High-Value"
        elif r["seasonal_concentration"] > 0.7:
            n = "Seasonal Gift Shopper"
        elif r["frequency"] > 0.3 and r["total_spend"] < 0:
            n = "High-Frequency Low-Value"
        elif r["recency_days"] > 0.3:
            n = "At-Risk / Lapsing"
        elif r["recency_days"] < -0.3 and r["frequency"] < 0:
            n = "Recent Occasional Buyer"
        else:
            n = "Steady Mid-Value"
        names[int(seg)] = n
    seen: dict[str, int] = {}
    for seg in sorted(names):
        base = names[seg]
        seen[base] = seen.get(base, 0) + 1
        if list(names.values()).count(base) > 1:
            names[seg] = f"{base} ({chr(64 + seen[base])})"
    return names


def profile_segments(df: pd.DataFrame, labels: np.ndarray, names: dict[int, str]) -> pd.DataFrame:
    """Summary statistics per segment."""
    d = df.assign(segment_id=labels)
    d["segment_label"] = d["segment_id"].map(names)
    cols = ["recency_days", "frequency", "total_spend", "avg_spend", "avg_basket_size", "return_rate",
            "seasonal_concentration", "discount_sensitivity", "engagement_score"]
    prof = d.groupby(["segment_id", "segment_label"])[cols].mean().round(3)
    prof.insert(0, "customers", d.groupby(["segment_id", "segment_label"]).size())
    prof.insert(1, "share_pct", (100 * prof["customers"] / len(d)).round(1))
    return prof.reset_index()


def run_segmentation(feats: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Fit K-Means + secondary algorithm and return assignments, profiles and quality metrics."""
    scfg = cfg["segmentation"]
    seed = cfg["project"]["seed"]
    Z, scaler = _prep(feats, cfg)

    ks = list(range(scfg["k_min"], scfg["k_max"] + 1))
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        inertia.append(float(km.inertia_))
        sil.append(float(silhouette_score(Z, km.labels_)))
    best_k = ks[int(np.argmax(sil))]
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(Z)
    km_labels = km.labels_

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[0].plot(ks, inertia, "o-")
    axs[0].set_title("Elbow (inertia)")
    axs[0].set_xlabel("k")
    axs[1].plot(ks, sil, "o-", color="darkorange")
    axs[1].axvline(best_k, ls="--", color="grey")
    axs[1].set_title(f"Silhouette (chosen k={best_k})")
    axs[1].set_xlabel("k")
    fig.tight_layout()
    fig.savefig(_fig(cfg, "kmeans_k_selection.png"), dpi=130)
    plt.close(fig)

    secondary: dict[str, Any] = {"algorithm": scfg["secondary_algorithm"]}
    if scfg["secondary_algorithm"] == "gmm":
        bics = {k: GaussianMixture(k, covariance_type="full", random_state=seed).fit(Z).bic(Z) for k in ks}
        best_c = min(bics, key=lambda k: bics[k])
        gm = GaussianMixture(best_c, covariance_type="full", random_state=seed).fit(Z)
        sec_labels = gm.predict(Z)
        secondary.update({"n_components": int(best_c), "bic": {int(k): float(v) for k, v in bics.items()}})
        sec_model: Any = gm
    else:
        nn = NearestNeighbors(n_neighbors=2 * Z.shape[1]).fit(Z)
        eps = float(np.percentile(nn.kneighbors(Z)[0][:, -1], 90))
        db = DBSCAN(eps=eps, min_samples=2 * Z.shape[1]).fit(Z)
        sec_labels = db.labels_
        secondary.update({"eps": eps, "n_clusters": int(len(set(sec_labels)) - (1 if -1 in sec_labels else 0)),
                          "noise_fraction": float((sec_labels == -1).mean())})
        sec_model = db
    ari = None
    try:
        from sklearn.metrics import adjusted_rand_score
        ari = float(adjusted_rand_score(km_labels, sec_labels))
    except Exception:  # noqa: BLE001
        pass
    metrics = {
        "kmeans": {"k": int(best_k), **_quality(Z, km_labels), "k_values": ks, "inertia": inertia, "silhouette_by_k": sil},
        "secondary": {**secondary, **_quality(Z, np.asarray(sec_labels)), "adjusted_rand_vs_kmeans": ari},
    }

    pca = PCA(n_components=2, random_state=seed).fit(Z)
    pcs = pca.transform(Z)
    metrics["pca_explained_variance"] = [float(v) for v in pca.explained_variance_ratio_]

    rng = np.random.default_rng(seed)
    sample = rng.choice(len(Z), size=min(scfg["hierarchical_sample"], len(Z)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    dendrogram(linkage(Z[sample], method="ward"), no_labels=True, truncate_mode="lastp", p=30, ax=ax)
    ax.set_title(f"Hierarchical (Ward) sanity check on a {len(sample)}-customer sample")
    fig.tight_layout()
    fig.savefig(_fig(cfg, "dendrogram_sample.png"), dpi=130)
    plt.close(fig)

    names = label_segments(feats, km_labels)
    profiles = profile_segments(feats, km_labels, names)
    tiers = scfg["value_tiers"]
    value_tier = pd.qcut(feats["historical_clv"].rank(method="first"), len(tiers), labels=tiers).astype(str)
    assign = pd.DataFrame({
        "customer_id": feats["customer_id"].to_numpy(), "kmeans_segment": km_labels,
        "segment_label": pd.Series(km_labels).map(names).to_numpy(), "secondary_cluster": np.asarray(sec_labels),
        "value_tier": value_tier.to_numpy(), "pca_x": pcs[:, 0], "pca_y": pcs[:, 1],
    })
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for lab, grp in assign.groupby("segment_label"):
        ax.scatter(grp["pca_x"], grp["pca_y"], s=6, alpha=0.6, label=lab)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Customer segments (PCA projection)")
    ax.legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(_fig(cfg, "segments_pca.png"), dpi=130)
    plt.close(fig)

    joblib.dump({"scaler": scaler, "kmeans": km, "secondary": sec_model, "pca": pca, "names": names,
                 "features": scfg["features"]}, artifacts_path(cfg, "segmentation", "segmentation.joblib"))
    logger.info("segmentation: k=%d silhouette=%.3f", best_k, metrics["kmeans"]["silhouette"])
    return {"assignments": assign, "profiles": profiles, "metrics": metrics}


def _fig(cfg: dict[str, Any], name: str) -> Any:
    p = path_of(cfg, "figures_dir") / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
