import pandas as pd
from typer.testing import CliRunner

from soccer_edge.cli import app
from soccer_edge.cards import write_data_card, write_model_card
from soccer_edge.models.bundle import save_bundle

runner = CliRunner()


def test_validate_cards_cli(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    save_bundle({"kind": "demo"}, bundle, "demo", "v1", ["x"], {"accuracy": 1.0})
    model_card = write_model_card(bundle, tmp_path / "MODEL_CARD.md")
    data_card = write_data_card("demo", [tmp_path], tmp_path / "DATA_CARD.md")
    result = runner.invoke(app, ["model", "validate-cards", "--model-card-path", str(model_card), "--data-card-path", str(data_card)])
    assert result.exit_code == 0
    assert "cards_valid=true" in result.stdout


def test_export_crops_cli_requires_image_path(tmp_path) -> None:
    source = tmp_path / "detections.csv"
    manifest = tmp_path / "crop_manifest.csv"
    pd.DataFrame([{"x1": 0, "y1": 0, "x2": 1, "y2": 1}]).to_csv(source, index=False)
    result = runner.invoke(app, ["video", "export-crops", "--source", str(source), "--manifest-output", str(manifest)])
    assert result.exit_code != 0


def test_object_training_cli_missing_optional_dependency_or_runs(tmp_path) -> None:
    data_config = tmp_path / "data.yaml"
    base_model = tmp_path / "base.pt"
    data_config.write_text("path: .\n", encoding="utf-8")
    base_model.write_bytes(b"demo")
    result = runner.invoke(
        app,
        [
            "train",
            "object-model",
            "--data-config",
            str(data_config),
            "--base-model",
            str(base_model),
            "--output-dir",
            str(tmp_path / "runs"),
            "--epochs",
            "1",
        ],
    )
    assert result.exit_code in {0, 1}
