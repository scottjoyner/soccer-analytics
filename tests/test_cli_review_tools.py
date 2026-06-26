import pandas as pd
from typer.testing import CliRunner

from soccer_edge.cli import app
from soccer_edge.models.bundle import save_bundle

runner = CliRunner()


def test_video_review_cli_commands(tmp_path) -> None:
    detections = tmp_path / "detections.csv"
    annotations = tmp_path / "annotations"
    low = tmp_path / "low.csv"
    pd.DataFrame(
        [{"frame_idx": 1, "class_name": "player", "confidence": 0.2, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
    ).to_csv(detections, index=False)
    result = runner.invoke(app, ["video", "export-annotations", "--source", str(detections), "--output-dir", str(annotations)])
    assert result.exit_code == 0
    assert (annotations / "1.txt").exists()
    result = runner.invoke(app, ["video", "sample-low-confidence", "--source", str(detections), "--output", str(low)])
    assert result.exit_code == 0
    assert low.exists()


def test_card_cli_commands(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    save_bundle({"kind": "demo"}, bundle, "demo", "v1", ["x"], {"accuracy": 1.0})
    model_card = tmp_path / "MODEL_CARD.md"
    data_card = tmp_path / "DATA_CARD.md"
    result = runner.invoke(app, ["model", "model-card", "--bundle-dir", str(bundle), "--output", str(model_card)])
    assert result.exit_code == 0
    assert model_card.exists()
    result = runner.invoke(app, ["model", "data-card", "--dataset-name", "demo", "--sources", str(tmp_path), "--output", str(data_card)])
    assert result.exit_code == 0
    assert data_card.exists()
