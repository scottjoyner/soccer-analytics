import pandas as pd
from typer.testing import CliRunner

from soccer_edge.cli import app

runner = CliRunner()


def test_tensor_samples_and_compare_commands(tmp_path) -> None:
    columns = [f"g{i}" for i in range(4)]
    source = tmp_path / "grid.csv"
    pd.DataFrame([{**{column: 0.0 for column in columns}, "label": 0}]).to_csv(source, index=False)
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
        ],
    )
    assert result.exit_code == 0
    assert tensor_out.exists()

    registry = tmp_path / "registry.csv"
    comparison = tmp_path / "comparison.csv"
    pd.DataFrame([{"name": "demo", "version": "v1", "metric_accuracy": 0.5}]).to_csv(registry, index=False)
    result = runner.invoke(app, ["model", "compare", "--registry", str(registry), "--output", str(comparison)])
    assert result.exit_code == 0
    assert comparison.exists()
