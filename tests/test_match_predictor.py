import pandas as pd

from soccer_edge.pipeline.match_predictor import (
    aggregate_video_features,
    build_prediction_dataset_multi,
    derive_winner_label,
    match_result_labels,
    train_match_predictor,
)


def test_derive_winner_label() -> None:
    assert derive_winner_label(2, 1) == 0
    assert derive_winner_label(0, 2) == 2
    assert derive_winner_label(1, 1) == 1


def test_match_result_labels_adds_winner() -> None:
    results = pd.DataFrame(
        [
            {"match_id": "m1", "home_score": 2, "away_score": 1},
            {"match_id": "m2", "home_score": 0, "away_score": 3},
            {"match_id": "m3", "home_score": 1, "away_score": 1},
        ]
    )
    labeled = match_result_labels(results)
    assert labeled["winner"].tolist() == [0, 2, 1]


def test_aggregate_video_features_handles_empty() -> None:
    features = aggregate_video_features(pd.DataFrame())
    assert features["n_player"] == 0
    assert features["n_ball"] == 0


def test_aggregate_video_features_counts_classes() -> None:
    detections = pd.DataFrame(
        [
            {"frame_idx": 0, "class_name": "player", "x1": 0, "y1": 0, "x2": 10, "y2": 20},
            {"frame_idx": 0, "class_name": "person", "x1": 0, "y1": 0, "x2": 10, "y2": 20},
            {"frame_idx": 1, "class_name": "ball", "x1": 5, "y1": 5, "x2": 10, "y2": 10},
        ]
    )
    features = aggregate_video_features(detections, video_width=100.0, video_height=100.0)
    assert features["n_player"] == 2
    assert features["n_ball"] == 1
    assert features["n_frames"] == 2
    assert 0.0 < features["ball_center_x"] < 1.0


def test_train_match_predictor_writes_bundles(tmp_path) -> None:
    results = pd.DataFrame(
        [
            {"match_id": "m1", "home_score": 2, "away_score": 1},
            {"match_id": "m2", "home_score": 0, "away_score": 3},
            {"match_id": "m3", "home_score": 1, "away_score": 1},
        ]
    )
    detections = {
        "m1": pd.DataFrame([{"frame_idx": 0, "class_name": "player", "x1": 0, "y1": 0, "x2": 10, "y2": 20}]),
        "m2": pd.DataFrame([{"frame_idx": 0, "class_name": "player", "x1": 0, "y1": 0, "x2": 10, "y2": 20}]),
        "m3": pd.DataFrame([{"frame_idx": 0, "class_name": "ball", "x1": 5, "y1": 5, "x2": 10, "y2": 10}]),
    }
    labeled = match_result_labels(results)
    dataset = build_prediction_dataset_multi(labeled, detections)
    paths = train_match_predictor(dataset, tmp_path)
    assert paths["winner_model"].exists()
    assert paths["home_score_model"].exists()
    assert paths["away_score_model"].exists()
    assert paths["predictions"].exists()
    preds = pd.read_csv(paths["predictions"])
    assert {"prob_0", "prob_1", "prob_2"}.issubset(preds.columns)
