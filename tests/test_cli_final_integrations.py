import pandas as pd
from typer.testing import CliRunner

from soccer_edge.cli import app

runner = CliRunner()


def test_annotation_audit_dataset_versions_and_object_eval_cli(tmp_path) -> None:
    annotations = tmp_path / "annotations.csv"
    audit_dir = tmp_path / "audit"
    versions = tmp_path / "versions.csv"
    object_eval = tmp_path / "object_eval.csv"
    pd.DataFrame(
        [
            {"frame_idx": 1, "class_name": "player", "split": "train", "status": "tp"},
            {"frame_idx": 2, "class_name": "ball", "split": "val", "status": "fn"},
        ]
    ).to_csv(annotations, index=False)

    result = runner.invoke(app, ["video", "audit-annotations", "--source", str(annotations), "--output-dir", str(audit_dir)])
    assert result.exit_code == 0
    assert (audit_dir / "by_class.csv").exists()

    result = runner.invoke(app, ["video", "dataset-versions", "--paths", str(annotations), "--output", str(versions)])
    assert result.exit_code == 0
    assert versions.exists()

    result = runner.invoke(app, ["model", "object-eval", "--source", str(annotations), "--output", str(object_eval)])
    assert result.exit_code == 0
    assert object_eval.exists()


def test_auto_data_card_and_source_catalog_cli(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    data_card = tmp_path / "DATA_CARD.md"
    source_catalog = tmp_path / "sources.csv"
    pd.DataFrame([{"rights_status": "owned", "class_name": "player"}]).to_csv(manifest, index=False)

    result = runner.invoke(
        app,
        ["model", "auto-data-card", "--dataset-name", "demo", "--manifests", str(manifest), "--output", str(data_card)],
    )
    assert result.exit_code == 0
    assert "Source catalog" in data_card.read_text(encoding="utf-8")

    result = runner.invoke(app, ["model", "source-catalog", "--output", str(source_catalog)])
    assert result.exit_code == 0
    assert source_catalog.exists()


def test_local_finetune_cli_help() -> None:
    result = runner.invoke(app, ["train", "local-finetune", "--help"])
    assert result.exit_code == 0
    assert "local fine-tuning" in result.stdout
