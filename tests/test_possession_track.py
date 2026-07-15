import pandas as pd

from soccer_edge.video.track_features import (
    POSSESSION_KEYS,
    build_match_track_features,
    build_possession_features,
)


def _det(frame, cls, x1, y1, x2, y2):
    return {"frame_idx": frame, "class_name": cls, "confidence": 0.9, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _controlled_frame(frame, n_players_near=1):
    rows = [_det(frame, "sports ball", 100, 100, 120, 120)]
    for i in range(n_players_near):
        rows.append(_det(frame, "person", 110 + i * 5, 110, 130 + i * 5, 130))
    return rows


def test_build_possession_features_empty() -> None:
    feats = build_possession_features(pd.DataFrame(columns=["frame_idx", "class_name", "x1", "y1", "x2", "y2"]))
    assert set(feats.keys()) == set(POSSESSION_KEYS)
    assert all(v == 0.0 for v in feats.values())


def test_build_possession_features_chains() -> None:
    rows = []
    for frame in range(5):  # frames 0..4 contiguous -> one chain of length 5
        rows.extend(_controlled_frame(frame, n_players_near=1))
    feats = build_possession_features(pd.DataFrame(rows))
    assert feats["n_possession_chains"] == 1.0
    assert feats["max_chain_frames"] == 5.0
    assert feats["mean_chain_frames"] == 5.0
    assert feats["possession_frame_rate"] == 1.0  # all 5 frames controlled


def test_build_possession_features_pressure() -> None:
    # one frame with 3 players near the ball -> high-pressure frame
    rows = _controlled_frame(0, n_players_near=3)
    feats = build_possession_features(pd.DataFrame(rows))
    assert feats["pressure_max"] == 3.0
    assert feats["pressure_high_rate"] == 1.0
    assert feats["contested_possession_rate"] == 1.0  # >=2 players near controlled ball


def test_build_possession_features_broken_chain() -> None:
    # frames 0 and 2 controlled (gap at 1) -> two chains of length 1
    rows = _controlled_frame(0) + _controlled_frame(2)
    feats = build_possession_features(pd.DataFrame(rows))
    assert feats["n_possession_chains"] == 2.0
    assert feats["max_chain_frames"] == 1.0


def test_match_track_features_includes_possession() -> None:
    rows = []
    for frame in range(3):
        rows.extend(_controlled_frame(frame, n_players_near=1))
    feats = build_match_track_features(pd.DataFrame(rows))
    for key in POSSESSION_KEYS:
        assert key in feats
    assert feats["n_possession_chains"] == 1.0
