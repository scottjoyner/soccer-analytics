import pandas as pd
from typer.testing import CliRunner

from soccer_edge.cli import app

runner = CliRunner()


def test_tensor_samples_group_and_markdown_report_commands(tmp_path) -> None:
    columns = [f"g{i}" for i in range(4)]
    source = tmp_path / "grid.csv"
    rows = []
    for match_id in ["m1", "m2"]:
        for label in [0, 1]:
            rows.append({**{column: 0.0 for column in columns}, "label": label, "match_id": match_id})
    pd.DataFrame(rows).to_csv(source, index=False)
    tensor_out = tmp_path / "samples.npz"
    result = runner.invoke(
        app,
        [
            "features",
            "tensor-samples",
            "--source",
            str(source),
            "--output",
            str(tensor_out),
            "--columns",
            ",".join(columns),
            "--channels",
            "1",
            "--height",
            "2",
            "--width",
            "2",
            "--sequence-length",
            "2",
            "--group",
            "match_id",
        ],
    )
    assert result.exit_code == 0
    assert tensor_out.exists()

    comparison = tmp_path / "comparison.csv"
    report = tmp_path / "comparison.md"
    pd.DataFrame([{"name": "demo", "metric_accuracy": 1.0}]).to_csv(comparison, index=False)
    result = runner.invoke(app, ["model", "compare-markdown", "--comparison", str(comparison), "--output", str(report)])
    assert result.exit_code == 0
    assert report.exists()
