import json

import pandas as pd
from typer.testing import CliRunner

from soccer_edge.cli import app

runner = CliRunner()


def test_contact_sheet_and_annotation_config_cli(tmp_path) -> None:
    crop_manifest = tmp_path / "crops.csv"
    review = tmp_path / "review.html"
    config = tmp_path / "data.yaml"
    pd.DataFrame([{"crop_path": "crop.jpg", "class_name": "player", "confidence": 0.7, "frame_idx": 1}]).to_csv(crop_manifest, index=False)
    result = runner.invoke(app, ["video", "contact-sheet", "--source", str(crop_manifest), "--output", str(review)])
    assert result.exit_code == 0
    assert review.exists()
    result = runner.invoke(
        app,
        [
            "video",
            "annotation-config",
            "--root",
            str(tmp_path),
            "--train-images",
            "images/train",
            "--val-images",
            "images/val",
            "--classes",
            "player,ball",
            "--output",
            str(config),
        ],
    )
    assert result.exit_code == 0
    assert config.exists()


def test_calibration_qa_cli(tmp_path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "pixel_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "pitch_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
            }
        ),
        encoding="utf-8",
    )
    csv_output = tmp_path / "qa.csv"
    svg_output = tmp_path / "qa.svg"
    result = runner.invoke(
        app,
        ["video", "calibration-qa", "--calibration", str(calibration), "--csv-output", str(csv_output), "--svg-output", str(svg_output)],
    )
    assert result.exit_code == 0
    assert csv_output.exists()
    assert svg_output.exists()


def test_export_frames_cli_missing_or_bad_file(tmp_path) -> None:
    missing = tmp_path / "missing.mp4"
    result = runner.invoke(app, ["video", "export-frames", "--input", str(missing), "--manifest-output", str(tmp_path / "frames.csv")])
    assert result.exit_code != 0
