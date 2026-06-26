import pandas as pd
from typer.testing import CliRunner

from soccer_edge.cli import app

runner = CliRunner()


def test_video_catalog_local_command(tmp_path) -> None:
    clip = tmp_path / "clip.mp4"
    manifest = tmp_path / "manifest.csv"
    clip.write_bytes(b"demo")
    result = runner.invoke(app, ["video", "catalog-local", "--root", str(tmp_path), "--output", str(manifest)])
    assert result.exit_code == 0
    assert manifest.exists()
    frame = pd.read_csv(manifest)
    assert frame.iloc[0]["rights_status"] == "owned"
