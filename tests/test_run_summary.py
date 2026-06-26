import pandas as pd

from soccer_edge.models.run_summary import write_run_summary


def test_write_run_summary(tmp_path) -> None:
    registry = tmp_path / "registry.csv"
    predictions = tmp_path / "predictions.csv"
    output = tmp_path / "summary"
    pd.DataFrame([{"name": "demo", "version": "v1", "metric_accuracy": 1.0}]).to_csv(registry, index=False)
    pd.DataFrame([{"label": 0, "prob_0": 0.9, "prob_1": 0.1}]).to_csv(predictions, index=False)
    paths = write_run_summary(registry, predictions, output)
    assert paths["comparison"].exists()
    assert paths["markdown"].exists()
    assert paths["review_metrics"].exists()
