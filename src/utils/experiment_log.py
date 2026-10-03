"""Lightweight experiment tracking: every model run is appended to a JSONL log.

The final comparison table is regenerated from this log, never typed by hand.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import artifacts_path


class ExperimentLogger:
    """Append-only run log stored under models_artifacts/experiment_log.jsonl."""

    def __init__(self, cfg: dict[str, Any], run_id: str) -> None:
        self.path: Path = artifacts_path(cfg, "experiment_log.jsonl")
        self.run_id = run_id

    def log(
        self,
        task: str,
        model: str,
        split: str,
        metrics: dict[str, Any],
        params: dict[str, Any] | None = None,
        train_time_sec: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append one run record."""
        record = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task": task,
            "model": model,
            "split": split,
            "params": params or {},
            "train_time_sec": None if train_time_sec is None else round(train_time_sec, 3),
            "metrics": metrics,
            "extra": extra or {},
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def read_log(cfg: dict[str, Any], run_id: str | None = None) -> pd.DataFrame:
    """Read the experiment log as a flat DataFrame (optionally filtered to one run)."""
    path = artifacts_path(cfg, "experiment_log.jsonl")
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows, sep=".")
    if run_id is not None:
        df = df[df["run_id"] == run_id]
    return df.reset_index(drop=True)


def reset_log(cfg: dict[str, Any]) -> None:
    """Remove the experiment log (start of a fresh full pipeline run)."""
    path = artifacts_path(cfg, "experiment_log.jsonl")
    if path.exists():
        path.unlink()
