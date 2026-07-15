import json

import pandas as pd

from soccer_edge.evaluation.promotion_metrics import (
    read_predictive_table,
    write_classification_predictive_metrics,
    write_predictive_metrics,
)


def test_cnn_single_metrics(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"match_accuracy": 0.50, "winner_brier": 0.62, "match_baseline_accuracy": 0.50}), encoding="utf-8")
    frame = read_predictive_table(metrics)
    assert list(frame.columns) == ["model", "accuracy", "brier", "baseline_accuracy", "split"]
    assert frame.iloc[0]["accuracy"] == 0.50
    assert frame.iloc[0]["brier"] == 0.62
    assert frame.iloc[0]["baseline_accuracy"] == 0.50


def test_cnn_repeated_cv_metrics(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"match_accuracy_mean": 0.489, "winner_brier_mean": 0.630, "match_baseline_accuracy_mean": 0.50}), encoding="utf-8")
    frame = read_predictive_table(metrics)
    assert frame.iloc[0]["accuracy"] == 0.489
    assert frame.iloc[0]["brier"] == 0.630
    assert frame.iloc[0]["baseline_accuracy"] == 0.50


def test_highlights_nested_metrics(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "v1_count_features": {"winner_accuracy_test": 0.55, "winner_brier_test": 0.30, "winner_accuracy_train": 0.60},
                "v2_track_features": {"winner_accuracy_test": 0.48, "winner_brier_test": 0.40, "winner_accuracy_train": 0.59},
            }
        ),
        encoding="utf-8",
    )
    frame = read_predictive_table(metrics)
    assert len(frame) == 2
    assert set(frame["model"]) == {"v1_count_features", "v2_track_features"}
    assert frame[frame["model"] == "v1_count_features"].iloc[0]["accuracy"] == 0.55


def test_write_predictive_metrics(tmp_path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"match_accuracy": 0.51, "winner_brier": 0.61, "match_baseline_accuracy": 0.50}), encoding="utf-8")
    out = tmp_path / "pred.csv"
    path = write_predictive_metrics(metrics, out, model_name="cnn-v1")
    assert path.exists()
    frame = pd.read_csv(path)
    assert frame.iloc[0]["model"] == "cnn-v1"
    assert frame.iloc[0]["accuracy"] == 0.51


def test_write_classification_predictive_metrics(tmp_path) -> None:
    from soccer_edge.models.metrics import ClassificationMetrics

    metrics = ClassificationMetrics(log_loss=0.7, brier_score=0.31, accuracy=0.62)
    out = tmp_path / "pred.csv"
    path = write_classification_predictive_metrics(metrics, out, model_name="tabular-v1")
    assert path.exists()
    frame = pd.read_csv(path)
    assert frame.iloc[0]["model"] == "tabular-v1"
    assert frame.iloc[0]["accuracy"] == 0.62
    assert frame.iloc[0]["brier"] == 0.31

