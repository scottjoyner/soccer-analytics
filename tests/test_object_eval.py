import pandas as pd
import pytest

from soccer_edge.object_eval import counts_from_status_rows, metrics_from_counts, object_eval_metrics, write_object_eval_metrics


def test_counts_from_status_rows() -> None:
    frame = pd.DataFrame(
        [
            {"class_name": "player", "status": "tp"},
            {"class_name": "player", "status": "fp"},
            {"class_name": "ball", "status": "fn"},
        ]
    )
    counts = counts_from_status_rows(frame)
    assert int(counts[counts["class_name"] == "player"].iloc[0]["tp"]) == 1


def test_metrics_from_counts() -> None:
    metrics = metrics_from_counts(pd.DataFrame([{"class_name": "player", "tp": 8, "fp": 2, "fn": 2}]))
    assert metrics.iloc[0]["precision"] == 0.8
    assert metrics.iloc[0]["recall"] == 0.8


def test_object_eval_metrics_from_status() -> None:
    metrics = object_eval_metrics(pd.DataFrame([{"class_name": "player", "status": "tp"}]))
    assert metrics.iloc[0]["f1"] == 1.0


def test_write_object_eval_metrics(tmp_path) -> None:
    source = tmp_path / "eval.csv"
    output = tmp_path / "metrics.csv"
    pd.DataFrame([{"class_name": "player", "status": "tp"}]).to_csv(source, index=False)
    path = write_object_eval_metrics(source, output)
    assert path.exists()


def test_metrics_from_counts_validates_columns() -> None:
    with pytest.raises(ValueError):
        metrics_from_counts(pd.DataFrame([{"class_name": "player"}]))
