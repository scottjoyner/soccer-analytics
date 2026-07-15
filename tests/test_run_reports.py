import pandas as pd

from soccer_edge.models.metrics import ClassificationMetrics
from soccer_edge.run_reports import to_plain, write_json, write_table


def test_to_plain_dataclass() -> None:
    payload = to_plain(ClassificationMetrics(log_loss=1.0, brier_score=0.5, accuracy=0.75, majority_baseline_accuracy=0.6))
    assert payload["accuracy"] == 0.75


def test_write_json(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    write_json({"accuracy": 1.0}, path)
    assert path.exists()
    assert "accuracy" in path.read_text(encoding="utf-8")


def test_write_table(tmp_path) -> None:
    path = tmp_path / "table.csv"
    write_table(pd.DataFrame([{"x": 1}]), path)
    assert path.exists()
    assert "x" in path.read_text(encoding="utf-8")
