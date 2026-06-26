import pandas as pd

from soccer_edge.models.comparison import build_model_comparison, write_model_comparison


def test_build_model_comparison_sorts_by_metric() -> None:
    registry = pd.DataFrame(
        [
            {"name": "a", "version": "v1", "metric_accuracy": 0.4},
            {"name": "b", "version": "v1", "metric_accuracy": 0.9},
        ]
    )
    comparison = build_model_comparison(registry)
    assert comparison.iloc[0]["name"] == "b"


def test_write_model_comparison(tmp_path) -> None:
    registry_path = tmp_path / "registry.csv"
    output_path = tmp_path / "comparison.csv"
    pd.DataFrame([{"name": "a", "version": "v1", "metric_accuracy": 0.4}]).to_csv(registry_path, index=False)
    path = write_model_comparison(registry_path, output_path)
    assert path.exists()
