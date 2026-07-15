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
    if isinstance(data, dict) and "_candidate_keys" not in data:
        flat = _candidate_row(data)
        if flat is not None:
            rows.append({"model": "model", **flat, "split": "test"})
        else:
            # Treat each top-level entry as a candidate sub-model. A flat dict with
            # extra scalar keys must not be silently misread as a single flat row.
            for model_name, sub in data.items():
                if not isinstance(sub, dict):
                    continue
                row = _candidate_row(sub)
                if row is not None:
                    rows.append({"model": str(model_name), **row, "split": "test"})
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


def write_classification_predictive_metrics(metrics, output_path: Path, model_name: str = "model", split: str = "eval") -> Path:
    """Write a promotion-gate predictive metrics CSV from a ClassificationMetrics object."""

    frame = pd.DataFrame(
        [
            {
                "model": model_name,
                "accuracy": float(metrics.accuracy),
                "brier": float(metrics.brier_score),
                "baseline_accuracy": float(metrics.majority_baseline_accuracy),
                "split": split,
            }
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(frame.to_csv(index=False), encoding="utf-8")
    return output_path
