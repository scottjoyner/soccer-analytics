from dataclasses import dataclass

import numpy as np
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
    validate_replay_frame(frame)
    if bucket_column not in frame.columns:
        raise ValueError(f"missing bucket column: {bucket_column}")

    rows: list[dict[str, object]] = []
    for bucket, group in frame.groupby(bucket_column):
        result = replay_predictions(group)
        rows.append(
            {
                "bucket": bucket,
                "row_count": result.row_count,
                "log_loss": result.metrics.log_loss,
                "brier_score": result.metrics.brier_score,
                "accuracy": result.metrics.accuracy,
            }
        )
    return pd.DataFrame(rows)
