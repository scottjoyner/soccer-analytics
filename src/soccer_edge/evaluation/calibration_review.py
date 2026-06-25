from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from soccer_edge.models.calibration import CalibrationReport, confidence_bins
from soccer_edge.models.metrics import ClassificationMetrics, score_classification
from soccer_edge.run_reports import write_json, write_table


@dataclass(frozen=True)
class CalibrationReview:
    row_count: int
    metrics: ClassificationMetrics
    calibration: CalibrationReport


def probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("prob_")]
    return sorted(columns, key=lambda item: int(item.split("_")[1]))


def build_calibration_review(frame: pd.DataFrame, label_column: str = "label", num_bins: int = 10) -> CalibrationReview:
    prob_columns = probability_columns(frame)
    if label_column not in frame.columns:
        raise ValueError(f"missing label column: {label_column}")
    if not prob_columns:
        raise ValueError("missing probability columns")
    probabilities = frame[prob_columns].to_numpy(dtype=float)
    labels = frame[label_column].to_numpy(dtype=int)
    return CalibrationReview(
        row_count=len(frame),
        metrics=score_classification(probabilities, labels),
        calibration=confidence_bins(probabilities, labels, num_bins=num_bins),
    )


def write_calibration_review(frame: pd.DataFrame, output_dir: Path, num_bins: int = 10) -> dict[str, Path]:
    review = build_calibration_review(frame, num_bins=num_bins)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    calibration_path = output_dir / "calibration.json"
    predictions_path = output_dir / "rows.csv"
    write_json(review.metrics, metrics_path)
    write_json(review.calibration, calibration_path)
    write_table(frame, predictions_path)
    return {"metrics": metrics_path, "calibration": calibration_path, "rows": predictions_path}
