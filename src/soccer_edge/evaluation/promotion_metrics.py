"""Bridge eval metrics.json outputs into the promotion-gate predictive metrics table.

The evaluation scripts (`scripts/evaluate_cnn.py`, `scripts/evaluate_highlights.py`)
emit a `metrics.json` whose schema differs per script. This module normalizes the
accuracy/brier fields those scripts report into a flat CSV the promotion gate
(`model promotion-gate --predictive-metrics`) can consume.
"""

import json
from pathlib import Path

import pandas as pd


def _candidate_row(metrics: dict) -> dict | None:
    """Extract an ``accuracy``/``brier``/``baseline_accuracy`` row from a metric dict."""

    if "match_accuracy_mean" in metrics:
        return {
            "accuracy": float(metrics["match_accuracy_mean"]),
            "brier": float(metrics.get("winner_brier_mean", 1.0)),
            "baseline_accuracy": float(metrics.get("match_baseline_accuracy_mean", 0.0)),
        }
    if "match_accuracy" in metrics:
        return {
            "accuracy": float(metrics["match_accuracy"]),
            "brier": float(metrics.get("winner_brier", 1.0)),
            "baseline_accuracy": float(metrics.get("match_baseline_accuracy", 0.0)),
        }
    if "winner_accuracy_test" in metrics:
        return {
            "accuracy": float(metrics["winner_accuracy_test"]),
            "brier": float(metrics.get("winner_brier_test", 1.0)),
            "baseline_accuracy": float(metrics.get("winner_accuracy_train", 0.0)),
        }
    return None


def read_predictive_table(metrics_json_path: Path) -> pd.DataFrame:
    data = json.loads(Path(metrics_json_path).read_text(encoding="utf-8"))
    rows: list[dict] = []
    if isinstance(data, dict) and data and all(isinstance(value, dict) for value in data.values()):
        for model_name, sub in data.items():
            row = _candidate_row(sub)
            if row is not None:
                rows.append({"model": model_name, **row, "split": "test"})
    else:
        row = _candidate_row(data)
        if row is not None:
            rows.append({"model": "model", **row, "split": "test"})
    if not rows:
        raise ValueError(f"no accuracy/brier fields found in {metrics_json_path}")
    return pd.DataFrame(rows, columns=["model", "accuracy", "brier", "baseline_accuracy", "split"])


def write_predictive_metrics(metrics_json_path: Path, output_path: Path, model_name: str | None = None) -> Path:
    frame = read_predictive_table(metrics_json_path)
    if model_name is not None and len(frame) == 1:
        frame = frame.copy()
        frame["model"] = model_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(frame.to_csv(index=False), encoding="utf-8")
    return output_path
