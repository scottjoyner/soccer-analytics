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
    assert "StatsBomb" in result.stdout


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
