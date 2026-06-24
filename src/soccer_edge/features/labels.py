from dataclasses import dataclass


@dataclass(frozen=True)
class TimedOutcome:
    match_id: str
    timestamp_seconds: float
    outcome_type: str
    team: str = ""


@dataclass(frozen=True)
class WindowLabel:
    match_id: str
    feature_timestamp_seconds: float
    label_name: str
    label_value: int
    label_window_seconds: float


def outcome_in_future_window(
    match_id: str,
    feature_timestamp_seconds: float,
    outcomes: list[TimedOutcome],
    outcome_type: str,
    window_seconds: float,
    label_name: str,
) -> WindowLabel:
    window_end = feature_timestamp_seconds + window_seconds
    value = any(
        outcome.match_id == match_id
        and outcome.outcome_type == outcome_type
        and feature_timestamp_seconds < outcome.timestamp_seconds <= window_end
        for outcome in outcomes
    )
    return WindowLabel(
        match_id=match_id,
        feature_timestamp_seconds=feature_timestamp_seconds,
        label_name=label_name,
        label_value=int(value),
        label_window_seconds=window_seconds,
    )


def assert_no_future_feature_rows(feature_timestamps: list[float], prediction_timestamp: float) -> None:
    future_rows = [timestamp for timestamp in feature_timestamps if timestamp > prediction_timestamp]
    if future_rows:
        raise ValueError("Feature rows include timestamps after prediction time")
