import pandas as pd

from soccer_edge.local_training_chain import run_local_training_chain


def test_run_local_training_chain(tmp_path) -> None:
    footage = tmp_path / "footage"
    footage.mkdir()
    (footage / "clip.mp4").write_bytes(b"demo")
    tabular = tmp_path / "training.csv"
    grid = tmp_path / "grid.csv"
    detections = tmp_path / "detections.csv"
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
    pd.DataFrame(
        [{"frame_idx": 1, "class_name": "player", "confidence": 0.2, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
    ).to_csv(detections, index=False)

    paths = run_local_training_chain(
        footage,
        tmp_path / "out",
        tabular,
        grid,
        ["speed_last", "pressure_last"],
        ["g0", "g1", "g2", "g3"],
        order_column="timestamp_seconds",
        rights_reference="personal-recording://self",
        detection_source=detections,
    )
    assert paths["manifest"].exists()
    assert paths["model"].exists()
    assert paths["tensor_samples"].exists()
    assert paths["model_card"].exists()
    assert paths["data_card"].exists()
    assert paths["low_confidence"].exists()
