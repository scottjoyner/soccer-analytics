import pandas as pd
import pytest

from soccer_edge.evaluation.replay import replay_predictions, score_by_bucket, score_by_group, score_standard_groups, validate_replay_frame


def test_replay_predictions() -> None:
    frame = pd.DataFrame(
        [
            {"match_id": "m1", "timestamp_seconds": 2.0, "label": 1, "prob_0": 0.1, "prob_1": 0.8, "prob_2": 0.1},
            {"match_id": "m1", "timestamp_seconds": 1.0, "label": 0, "prob_0": 0.7, "prob_1": 0.2, "prob_2": 0.1},
        ]
    )
    result = replay_predictions(frame)
    assert result.row_count == 2
    assert result.metrics.accuracy == 1.0


def test_score_by_bucket() -> None:
    frame = pd.DataFrame(
        [
            {"match_id": "m1", "timestamp_seconds": 1.0, "label": 0, "prob_0": 0.7, "prob_1": 0.2, "prob_2": 0.1, "bucket": "a"},
            {"match_id": "m2", "timestamp_seconds": 1.0, "label": 1, "prob_0": 0.1, "prob_1": 0.8, "prob_2": 0.1, "bucket": "b"},
        ]
    )
    scored = score_by_bucket(frame, "bucket")
    assert set(scored["bucket"]) == {"a", "b"}


def test_score_by_group_multiple_columns() -> None:
    frame = pd.DataFrame(
        [
            {"match_id": "m1", "timestamp_seconds": 1.0, "label": 0, "prob_0": 0.7, "prob_1": 0.2, "prob_2": 0.1, "league": "cup", "team": "A"},
            {"match_id": "m2", "timestamp_seconds": 1.0, "label": 1, "prob_0": 0.1, "prob_1": 0.8, "prob_2": 0.1, "league": "cup", "team": "B"},
        ]
    )
    scored = score_by_group(frame, ["league", "team"])
    assert set(scored["team"]) == {"A", "B"}


def test_score_standard_groups() -> None:
    frame = pd.DataFrame(
        [
            {"match_id": "m1", "timestamp_seconds": 1.0, "label": 0, "prob_0": 0.7, "prob_1": 0.2, "prob_2": 0.1, "league": "cup", "team": "A", "time_window": "early"},
        ]
    )
    outputs = score_standard_groups(frame)
    assert {"league", "team", "time_window", "combined"}.issubset(outputs)


def test_validate_replay_frame_missing_column() -> None:
    with pytest.raises(ValueError):
        validate_replay_frame(pd.DataFrame([{"match_id": "m1"}]))
