from typer.testing import CliRunner

from soccer_edge.cli import app

runner = CliRunner()


def test_doctor_command() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "env=dev" in result.stdout


def test_ingest_statsbomb_command() -> None:
    result = runner.invoke(app, ["ingest", "statsbomb", "--path", "data/raw/statsbomb"])
    assert result.exit_code == 0
    assert "loaded" in result.stdout


def test_video_plan_command(tmp_path) -> None:
    licensed_root = tmp_path / "licensed"
    licensed_root.mkdir()
    clip_path = licensed_root / "clip.mp4"
    clip_path.touch()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "video_id,match_id,clip_type,local_path,rights_status\n"
        f"clip,match,full_match,{clip_path},licensed\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["video", "plan", "--manifest", str(manifest), "--licensed-root", str(licensed_root)],
    )
    assert result.exit_code == 0
    assert "processable=1" in result.stdout


def test_features_build_command(tmp_path) -> None:
    output_dir = tmp_path / "state_tables"
    result = runner.invoke(app, ["features", "build", "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    assert (output_dir / "ball_states.parquet").exists()


def test_model_save_demo_command(tmp_path) -> None:
    output_dir = tmp_path / "bundle"
    result = runner.invoke(app, ["model", "save-demo", "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metadata.json").exists()


def test_model_evaluate_command(tmp_path) -> None:
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "match_id,timestamp_seconds,label,prob_0,prob_1,prob_2\n"
        "m1,1.0,0,0.8,0.1,0.1\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["model", "evaluate", "--predictions", str(predictions)])
    assert result.exit_code == 0
    assert "row_count=1" in result.stdout
