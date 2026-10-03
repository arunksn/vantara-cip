"""Programmatic generation of the architecture, ER and workflow diagrams (PNG, no external tools needed)."""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src.utils.config import path_of  # noqa: E402
from src.utils.db import Base  # noqa: E402


def _box(ax: Any, x: float, y: float, w: float, h: float, text: str, color: str = "#dbe9f6", fs: int = 8) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="#33506b", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def _arrow(ax: Any, x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": "#33506b", "lw": 1.3})
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", fontsize=7, color="#33506b")


def architecture_diagram(path: Any) -> None:
    """System architecture: data -> pipeline -> artifacts -> API / DB -> dashboard."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.set_title("System architecture", fontsize=12, fontweight="bold")
    _box(ax, 0.2, 2.3, 1.7, 1.0, "UCI Online Retail II\n(ucimlrepo / xlsx)", "#f6e8c8")
    _box(ax, 2.4, 2.3, 2.0, 1.0, "Training pipeline\nclean, validate, features,\nmodels, segmentation", "#dbe9f6")
    _box(ax, 2.4, 0.5, 2.0, 1.0, "models_artifacts/\nmodels, transformer,\nmetrics, SHAP", "#e3f2dc")
    _box(ax, 5.0, 3.6, 2.0, 1.0, "Batch scoring job\n(src.models.batch_score)", "#dbe9f6")
    _box(ax, 7.6, 3.6, 2.0, 1.0, "PostgreSQL\ncustomers, transactions,\npredictions, segments", "#f2dede")
    _box(ax, 5.0, 1.6, 2.0, 1.0, "FastAPI service\n/predict/customer\n/predict/batch /health\n/model/metadata /explain", "#dbe9f6")
    _box(ax, 8.2, 1.2, 2.5, 1.2, "Streamlit dashboard\nsegments, churn leaderboard,\nrevenue + forecast, SHAP/LIME", "#e8dff5")
    _box(ax, 5.0, 0.2, 2.0, 0.8, "Docker Compose\n(db, init, api, dashboard)", "#eeeeee")
    _arrow(ax, 1.9, 2.8, 2.4, 2.8)
    _arrow(ax, 3.4, 2.3, 3.4, 1.5, "persist")
    _arrow(ax, 4.4, 1.2, 5.0, 2.0, "load")
    _arrow(ax, 4.4, 3.0, 5.0, 3.9)
    _arrow(ax, 7.0, 4.1, 7.6, 4.1, "write")
    _arrow(ax, 8.6, 3.6, 9.2, 2.4, "read")
    _arrow(ax, 7.0, 2.1, 8.2, 1.9, "REST/JSON")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def er_diagram(path: Any) -> None:
    """ER diagram generated from the SQLAlchemy models (cannot drift from the schema)."""
    tables = list(Base.metadata.sorted_tables)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.4)
    ax.axis("off")
    ax.set_title("Entity-relationship diagram", fontsize=12, fontweight="bold")
    pos = {"customers": (4.6, 3.2), "transactions": (0.2, 0.6), "predictions": (8.6, 3.5), "segments": (8.6, 0.2)}
    centers: dict[str, tuple[float, float]] = {}
    for t in tables:
        x, y = pos[t.name]
        h = 0.34 * (len(t.columns) + 1)
        w = 3.1
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc="#fdfdfd", ec="#33506b", lw=1.2))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.34), w, 0.34, boxstyle="round,pad=0.02", fc="#bcd4ea", ec="#33506b"))
        ax.text(x + w / 2, y + h - 0.17, t.name, ha="center", va="center", fontsize=9, fontweight="bold")
        for i, c in enumerate(t.columns):
            tag = "PK,FK " if (c.primary_key and c.foreign_keys) else ("PK " if c.primary_key else ("FK " if c.foreign_keys else "   "))
            ax.text(x + 0.1, y + h - 0.34 - 0.3 * (i + 0.7), f"{tag}{c.name}: {str(c.type).split('(')[0]}", fontsize=6.5,
                    family="monospace")
        centers[t.name] = (x + w / 2, y + h / 2)
    for t in tables:
        for fk in t.foreign_keys:
            a, b = centers[t.name], centers[fk.column.table.name]
            ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": "#b03030", "lw": 1.2})
    ax.text(6, 8.6, "arrows: foreign key -> referenced table (many-to-one)", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def workflow_diagram(path: Any) -> None:
    """ML workflow diagram: stages from raw data to monitoring."""
    steps = ["Raw data\n(2 sheets)", "Validate +\nclean", "Point-in-time\nfeatures + labels", "Stratified\nsplit 70/15/15",
             "Train + tune\n(5-fold CV)", "Evaluate once\non test set", "Select + explain\n(SHAP/LIME)", "Segment +\nscore",
             "Serve: API,\nDB, dashboard"]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("ML workflow", fontsize=12, fontweight="bold")
    for i, s in enumerate(steps):
        x = 0.1 + i * 1.47
        _box(ax, x, 1.0, 1.25, 1.0, s, "#dbe9f6" if i < 7 else "#e3f2dc", fs=7)
        if i:
            _arrow(ax, x - 0.2, 1.5, x, 1.5)
    ax.text(6.7, 0.35, "Test set is used once for reporting. Class imbalance is handled on training data only. "
            "Scalers/encoders are fitted on training data and persisted.", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_diagrams(cfg: dict[str, Any]) -> None:
    """Write all three diagrams to docs/."""
    d = path_of(cfg, "docs_dir")
    d.mkdir(parents=True, exist_ok=True)
    architecture_diagram(d / "architecture_diagram.png")
    er_diagram(d / "er_diagram.png")
    workflow_diagram(d / "workflow_diagram.png")
