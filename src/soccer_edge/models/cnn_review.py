from pathlib import Path

import pandas as pd

from soccer_edge.evaluation.calibration_review import write_calibration_review
from soccer_edge.models.cnn_predict import export_cnn_bundle_predictions


def write_cnn_calibration_review(
    bundle_dir: Path,
    source: Path,
    output_dir: Path,
    batch_size: int = 8,
    num_bins: int = 10,
) -> dict[str, Path]:
    predictions_path = output_dir / "cnn_predictions.csv"
    export_cnn_bundle_predictions(bundle_dir=bundle_dir, npz_path=source, output=predictions_path, batch_size=batch_size)
    frame = pd.read_csv(predictions_path)
    paths = write_calibration_review(frame, output_dir, num_bins=num_bins)
    paths["predictions"] = predictions_path
    return paths
