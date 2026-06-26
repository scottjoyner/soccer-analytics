import pandas as pd
from typer.testing import CliRunner

from soccer_edge.cli import app
from soccer_edge.models.simple_classifier import fit_simple_classifier

runner = CliRunner()


def test_model_predict_and_registry_summary_commands(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"x": 0.0, "label": 0},
            {"x": 1.0, "label": 1},
            {"x": 2.0, "label": 1},
            {"x": -1.0, "label": 0},
        ]
    )
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)
    bundle_dir = tmp_path / "bundle"
    fit_simple_classifier(frame, ["x"], "label", bundle_dir)

    predictions = tmp_path / "predictions.csv"
    result = runner.invoke(
        app,
        ["model", "predict", "--bundle-dir", str(bundle_dir), "--source", str(source), "--output", str(predictions)],
    )
    assert result.exit_code == 0
    assert predictions.exists()
    assert "prob_0" in pd.read_csv(predictions).columns

    summary = tmp_path / "summary.csv"
    result = runner.invoke(app, ["model", "registry-summary", "--root-dir", str(tmp_path), "--output", str(summary)])
    assert result.exit_code == 0
    assert summary.exists()
