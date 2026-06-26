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


def test_ingest_metrica_command(tmp_path) -> None:
    (tmp_path / "RawEvents.csv").write_text("event_id,type\n1,pass\n", encoding="utf-8")
    result = runner.invoke(app, ["ingest", "metrica", "--path", str(tmp_path)])
    assert result.exit_code == 0
    assert "loaded" in result.stdout


def test_ingest_soccernet_command(tmp_path) -> None:
    (tmp_path / "labels.json").write_text('[{"label": "pass"}]', encoding="utf-8")
    result = runner.invoke(app, ["ingest", "soccernet", "--path", str(tmp_path)])
    assert result.exit_code == 0
    assert "loaded" in result.stdout


def test_ingest_write_processed_command(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "processed"
    source.mkdir()
    (source / "RawEvents.csv").write_text("event_id,type\n1,pass\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["ingest", "write-processed", "--source", str(source), "--output-dir", str(output), "--source-type", "metrica"],
    )
    assert result.exit_code == 0
    assert (output / "metrica_events.parquet").exists()


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


def test_video_process_command(tmp_path) -> None:
    clip = tmp_path / "clip.mp4"
    output = tmp_path / "video_out"
    clip.write_bytes(b"demo")
    result = runner.invoke(app, ["video", "process", "--input", str(clip), "--output-dir", str(output), "--frame-count", "1"])
    assert result.exit_code == 0
    assert (output / "detections.parquet").exists()


def test_features_build_command(tmp_path) -> None:
    output_dir = tmp_path / "state_tables"
    result = runner.invoke(app, ["features", "build", "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    assert (output_dir / "ball_states.parquet").exists()


def test_features_prematch_command(tmp_path) -> None:
    matches = tmp_path / "matches.csv"
    output = tmp_path / "prematch.parquet"
    matches.write_text("match_id,home_team,away_team\nm1,A,B\n", encoding="utf-8")
    result = runner.invoke(app, ["features", "prematch", "--matches", str(matches), "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()


def test_features_inplay_command(tmp_path) -> None:
    source = tmp_path / "inplay.csv"
    output = tmp_path / "inplay.parquet"
    source.write_text("match_id,timestamp_seconds,speed\nm1,1.0,1.0\nm1,2.0,3.0\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["features", "inplay", "--source", str(source), "--output", str(output), "--columns", "speed"],
    )
    assert result.exit_code == 0
    assert output.exists()


def test_model_save_demo_and_registry_commands(tmp_path) -> None:
    output_dir = tmp_path / "bundle"
    result = runner.invoke(app, ["model", "save-demo", "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    assert (output_dir / "model.joblib").exists()
    registry = tmp_path / "registry.csv"
    result = runner.invoke(app, ["model", "registry", "--root-dir", str(tmp_path), "--output", str(registry)])
    assert result.exit_code == 0
    assert registry.exists()


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


def test_model_calibration_review_command(tmp_path) -> None:
    predictions = tmp_path / "predictions.csv"
    output_dir = tmp_path / "calibration"
    predictions.write_text("label,prob_0,prob_1,prob_2\n0,0.8,0.1,0.1\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["model", "calibration-review", "--predictions", str(predictions), "--output-dir", str(output_dir)],
    )
    assert result.exit_code == 0
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "calibration.json").exists()
