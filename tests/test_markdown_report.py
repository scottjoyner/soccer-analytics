import pandas as pd

from soccer_edge.models.markdown_report import dataframe_to_markdown, write_model_markdown_report


def test_dataframe_to_markdown() -> None:
    text = dataframe_to_markdown(pd.DataFrame([{"name": "demo", "accuracy": 1.0}]))
    assert "demo" in text
    assert "accuracy" in text


def test_write_model_markdown_report(tmp_path) -> None:
    comparison = tmp_path / "comparison.csv"
    output = tmp_path / "report.md"
    pd.DataFrame([{"name": "demo", "accuracy": 1.0}]).to_csv(comparison, index=False)
    path = write_model_markdown_report(comparison, output)
    assert path.exists()
    assert "Model Comparison Report" in path.read_text(encoding="utf-8")
