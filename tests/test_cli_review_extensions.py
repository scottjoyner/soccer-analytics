import pandas as pd
from typer.testing import CliRunner

from soccer_edge.cli import app

runner = CliRunner()


def test_merge_corrections_and_object_confusion_cli(tmp_path) -> None:
    base = tmp_path / "base.csv"
    corrections = tmp_path / "corrections.csv"
    corrected = tmp_path / "corrected.csv"
    eval_rows = tmp_path / "eval.csv"
    matrix = tmp_path / "matrix.csv"
    svg = tmp_path / "matrix.svg"
    pd.DataFrame([{"crop_path": "a.jpg", "class_name": "player"}]).to_csv(base, index=False)
    pd.DataFrame([{"crop_path": "a.jpg", "review_action": "correct", "corrected_class_name": "ball"}]).to_csv(corrections, index=False)
    result = runner.invoke(app, ["video", "merge-corrections", "--base", str(base), "--corrections", str(corrections), "--output", str(corrected)])
    assert result.exit_code == 0
    assert pd.read_csv(corrected).iloc[0]["class_name"] == "ball"

    pd.DataFrame([{"actual_class": "player", "predicted_class": "player"}, {"actual_class": "ball", "predicted_class": "player"}]).to_csv(eval_rows, index=False)
    result = runner.invoke(app, ["model", "object-confusion", "--source", str(eval_rows), "--table-output", str(matrix), "--svg-output", str(svg)])
    assert result.exit_code == 0
    assert matrix.exists()
    assert svg.exists()


def test_local_finetune_dry_run_plan_cli(tmp_path) -> None:
    clip = tmp_path / "clip.mp4"
    model = tmp_path / "model.pt"
    plan = tmp_path / "plan.sh"
    clip.write_bytes(b"demo")
    model.write_bytes(b"demo")
    result = runner.invoke(
        app,
        [
            "train",
            "local-finetune",
            "--input",
            str(clip),
            "--object-model-path",
            str(model),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run-plan",
            str(plan),
        ],
    )
    assert result.exit_code == 0
    assert plan.exists()
    assert "soccer-edge video export-frames" in plan.read_text(encoding="utf-8")


def test_data_card_version_cli(tmp_path) -> None:
    source = tmp_path / "source.csv"
    card = tmp_path / "DATA_CARD.md"
    pd.DataFrame([{"x": 1}]).to_csv(source, index=False)
    result = runner.invoke(
        app,
        ["model", "data-card", "--dataset-name", "demo", "--sources", str(source), "--output", str(card), "--version-paths", str(source)],
    )
    assert result.exit_code == 0
    assert "Dataset version ID:" in card.read_text(encoding="utf-8")
