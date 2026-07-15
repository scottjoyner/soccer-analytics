import numpy as np
import pandas as pd

from soccer_edge.video.track_features import (
    FEATURE_COLUMNS,
    build_match_track_features,
    build_track_dataset,
    evaluate_match_predictor,
)


def _det(frame, cls, x1, y1, x2, y2):
    return {"frame_idx": frame, "class_name": cls, "confidence": 0.9, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_build_match_track_features_basic() -> None:
    rows = [
        _det(0, "sports ball", 100, 100, 120, 120),
        _det(0, "person", 110, 110, 130, 140),
        _det(1, "sports ball", 200, 200, 220, 220),
        _det(1, "person", 210, 210, 230, 240),
    ]
    feats = build_match_track_features(pd.DataFrame(rows))
    assert feats["n_frames"] == 2
    assert feats["ball_rate"] == 1.0
    assert 0.0 < feats["player_ball_min_dist"] < 1.0
    assert feats["contested_rate"] == 1.0


def test_build_match_track_features_empty() -> None:
    feats = build_match_track_features(pd.DataFrame(columns=["frame_idx", "class_name", "x1", "y1", "x2", "y2"]))
    assert feats["n_frames"] == 0
    assert feats["ball_rate"] == 0.0


def test_build_track_dataset_labels() -> None:
    results = pd.DataFrame([{"match_id": "M1", "home_score": 2, "away_score": 1}])
    dets = pd.DataFrame([
        _det(0, "sports ball", 100, 100, 120, 120),
        _det(0, "person", 110, 110, 130, 140),
        _det(1, "person", 300, 300, 320, 330),
    ])
    ds = build_track_dataset(results, {"M1": dets})
    assert len(ds) == 1
    assert ds.iloc[0]["winner"] == 0  # home win
    assert ds.iloc[0]["home_score"] == 2


def test_evaluate_match_predictor_returns_metrics() -> None:
    rng = np.random.default_rng(0)
    n = 60
    frame = pd.DataFrame(
        {
            "ball_rate": rng.random(n),
            "person_frame_rate": rng.random(n),
            "player_ball_min_dist": rng.random(n),
            "players_near_ball_mean": rng.random(n),
            "contested_rate": rng.random(n),
            "ball_movement": rng.random(n),
            "player_box_area_mean": rng.random(n),
            "winner": rng.integers(0, 3, n),
            "home_score": rng.integers(0, 5, n),
            "away_score": rng.integers(0, 5, n),
        }
    )
    metrics = evaluate_match_predictor(frame, feature_columns=FEATURE_COLUMNS)
    for key in [
        "n_train",
        "n_test",
        "winner_accuracy_train",
        "winner_accuracy_test",
        "winner_brier_test",
        "home_score_mse_test",
        "away_score_mse_test",
    ]:
        assert key in metrics
    assert metrics["n_train"] + metrics["n_test"] == n
