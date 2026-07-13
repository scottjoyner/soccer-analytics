import pandas as pd
from typer.testing import CliRunner

from soccer_edge.cli import app

runner = CliRunner()


def test_attach_frame_images_and_split_annotations_cli(tmp_path) -> None:
    detections = tmp_path / "detections.csv"
    frames = tmp_path / "frames.csv"
    joined = tmp_path / "joined.csv"
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    pd.DataFrame([{"frame_idx": 1, "class_name": "player"}, {"frame_idx": 2, "class_name": "ball"}]).to_csv(detections, index=False)
    pd.DataFrame([{"frame_idx": 1, "image_path": "f1.jpg"}, {"frame_idx": 2, "image_path": "f2.jpg"}]).to_csv(frames, index=False)
    result = runner.invoke(
        app,
        ["video", "attach-frame-images", "--detections", str(detections), "--frame-manifest", str(frames), "--output", str(joined)],
    )
    assert result.exit_code == 0
    assert pd.read_csv(joined).iloc[0]["image_path"] == "f1.jpg"
    result = runner.invoke(
        app,
        ["video", "split-annotations", "--source", str(joined), "--train-output", str(train), "--val-output", str(val), "--train-fraction", "0.5"],
    )
    assert result.exit_code == 0
    assert train.exists()
    assert val.exists()


def test_calibration_summary_cli(tmp_path) -> None:
    source = tmp_path / "calibration.csv"
    output = tmp_path / "calibration.md"
    pd.DataFrame([{"error_m": 0.0}, {"error_m": 1.0}]).to_csv(source, index=False)
    result = runner.invoke(app, ["video", "calibration-summary", "--source", str(source), "--output", str(output)])
    assert result.exit_code == 0
    assert "Mean error" in output.read_text(encoding="utf-8")
