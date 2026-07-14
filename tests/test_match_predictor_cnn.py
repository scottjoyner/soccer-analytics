import pandas as pd

from soccer_edge.pipeline.match_predictor import (
    build_match_grid_table_multi,
    frame_occupancy_grid,
    match_result_labels,
    train_match_predictor_cnn,
)


def test_frame_occupancy_grid_shape_and_counts() -> None:
    detections = pd.DataFrame(
        [
            {"frame_idx": 0, "class_name": "player", "x1": 0, "y1": 0, "x2": 100, "y2": 100},
            {"frame_idx": 0, "class_name": "ball", "x1": 50, "y1": 50, "x2": 60, "y2": 60},
        ]
    )
    grid = frame_occupancy_grid(detections, channels=3, height_bins=4, width_bins=4, video_width=200, video_height=200)
    assert grid.shape == (3 * 4 * 4,)
    assert grid.sum() >= 2.0


def test_train_match_predictor_cnn_produces_npz_and_model(tmp_path) -> None:
    results = pd.DataFrame(
        [
            {"match_id": "m1", "home_score": 2, "away_score": 1},
            {"match_id": "m2", "home_score": 0, "away_score": 3},
        ]
    )
    det_m1 = pd.DataFrame(
        [
            {"frame_idx": 0, "class_name": "player", "x1": 0, "y1": 0, "x2": 50, "y2": 50},
            {"frame_idx": 1, "class_name": "ball", "x1": 10, "y1": 10, "x2": 20, "y2": 20},
        ]
    )
    det_m2 = pd.DataFrame(
        [
            {"frame_idx": 0, "class_name": "player", "x1": 30, "y1": 30, "x2": 80, "y2": 80},
            {"frame_idx": 1, "class_name": "player", "x1": 5, "y1": 5, "x2": 15, "y2": 15},
        ]
    )
    labeled = match_result_labels(results)
    grid_table = build_match_grid_table_multi(labeled, {"m1": det_m1, "m2": det_m2})
    assert "match_id" in grid_table.columns
    assert "winner" in grid_table.columns
    assert (grid_table["match_id"] == "m1").any()

    paths = train_match_predictor_cnn(
        grid_table,
        tmp_path,
        channels=3,
        height_bins=4,
        width_bins=4,
        sequence_length=2,
        device="cpu",
        epochs=1,
        batch_size=2,
        hidden_size=16,
    )
    assert paths["npz"].exists()
    assert paths["model"].exists()
    assert paths["metadata"].exists()
