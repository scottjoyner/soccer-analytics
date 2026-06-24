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
