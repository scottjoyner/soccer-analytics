from dataclasses import dataclass

import pandas as pd

from soccer_edge.models.metrics import ClassificationMetrics, score_classification


@dataclass(frozen=True)
class ReplayResult:
    row_count: int
    metrics: ClassificationMetrics


def validate_replay_frame(frame: pd.DataFrame) -> None:
    required = {"match_id", "timestamp_seconds", "label", "prob_0", "prob_1", "prob_2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")


def replay_predictions(frame: pd.DataFrame) -> ReplayResult:
    validate_replay_frame(frame)
    ordered = frame.sort_values(["match_id", "timestamp_seconds"]).reset_index(drop=True)
    probabilities = ordered[["prob_0", "prob_1", "prob_2"]].to_numpy(dtype=float)
    labels = ordered["label"].to_numpy(dtype=int)
    return ReplayResult(row_count=len(ordered), metrics=score_classification(probabilities, labels))


def score_by_bucket(frame: pd.DataFrame, bucket_column: str) -> pd.DataFrame:
    return score_by_group(frame, [bucket_column])


def score_by_group(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    validate_replay_frame(frame)
    if not group_columns:
        raise ValueError("group_columns cannot be empty")
    missing = [column for column in group_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing group columns: {missing}")

    rows: list[dict[str, object]] = []
    group_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for key, group in frame.groupby(group_key):
        if not isinstance(key, tuple):
            key = (key,)
        result = replay_predictions(group)
        row: dict[str, object] = {column: value for column, value in zip(group_columns, key, strict=True)}
        row.update(
            {
                "row_count": result.row_count,
                "log_loss": result.metrics.log_loss,
                "brier_score": result.metrics.brier_score,
                "accuracy": result.metrics.accuracy,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def score_standard_groups(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for column in ["league", "team", "time_window"]:
        if column in frame.columns:
            outputs[column] = score_by_group(frame, [column])
    available_combo = [column for column in ["league", "team", "time_window"] if column in frame.columns]
    if len(available_combo) >= 2:
        outputs["combined"] = score_by_group(frame, available_combo)
    return outputs
