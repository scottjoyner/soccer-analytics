import pandas as pd
from typer.testing import CliRunner

from soccer_edge.cli import app

runner = CliRunner()


def test_train_local_chain_command(tmp_path) -> None:
    footage = tmp_path / "footage"
    footage.mkdir()
    (footage / "clip.mp4").write_bytes(b"demo")
    tabular = tmp_path / "training.csv"
    grid = tmp_path / "grid.csv"
    pd.DataFrame(
        [
            {"speed_last": 0.0, "pressure_last": 0.1, "label": 0},
            {"speed_last": 1.0, "pressure_last": 0.9, "label": 1},
            {"speed_last": 0.2, "pressure_last": 0.1, "label": 0},
            {"speed_last": 0.8, "pressure_last": 0.9, "label": 1},
        ]
    ).to_csv(tabular, index=False)
    pd.DataFrame(
        [
            {"match_id": "m1", "timestamp_seconds": 1.0, "g0": 0.0, "g1": 0.0, "g2": 0.0, "g3": 0.0, "label": 0},
            {"match_id": "m1", "timestamp_seconds": 2.0, "g0": 1.0, "g1": 0.0, "g2": 0.0, "g3": 0.0, "label": 1},
        ]
    ).to_csv(grid, index=False)
    output = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "train",
            "local-chain",
            "--footage-root",
            str(footage),
            "--tabular-source",
            str(tabular),
            "--grid-source",
            str(grid),
            "--output-dir",
            str(output),
            "--tabular-columns",
            "speed_last,pressure_last",
            "--grid-columns",
            "g0,g1,g2,g3",
            "--order",
            "timestamp_seconds",
        ],
    )
    assert result.exit_code == 0
    assert (output / "predictions.csv").exists()
    assert (output / "MODEL_CARD.md").exists()
