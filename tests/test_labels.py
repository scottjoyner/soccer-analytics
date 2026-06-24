import pytest

from soccer_edge.features.labels import TimedOutcome, assert_no_future_feature_rows, outcome_in_future_window


def test_outcome_in_future_window_is_leakage_safe() -> None:
    outcomes = [TimedOutcome(match_id="match", timestamp_seconds=130.0, outcome_type="score")]
    label = outcome_in_future_window(
        match_id="match",
        feature_timestamp_seconds=100.0,
        outcomes=outcomes,
        outcome_type="score",
        window_seconds=60.0,
        label_name="score_next_60s",
    )
    assert label.label_value == 1


def test_outcome_at_feature_time_is_not_future() -> None:
    outcomes = [TimedOutcome(match_id="match", timestamp_seconds=100.0, outcome_type="score")]
    label = outcome_in_future_window(
        match_id="match",
        feature_timestamp_seconds=100.0,
        outcomes=outcomes,
        outcome_type="score",
        window_seconds=60.0,
        label_name="score_next_60s",
    )
    assert label.label_value == 0


def test_assert_no_future_feature_rows() -> None:
    with pytest.raises(ValueError):
        assert_no_future_feature_rows([10.0, 20.0, 30.0], prediction_timestamp=25.0)
