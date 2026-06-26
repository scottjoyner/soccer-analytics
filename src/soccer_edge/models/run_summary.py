from pathlib import Path

import pandas as pd

from soccer_edge.evaluation.calibration_review import write_calibration_review
from soccer_edge.models.comparison import write_model_comparison
from soccer_edge.models.markdown_report import write_model_markdown_report


def write_run_summary(
    registry_path: Path,
    predictions_path: Path,
    output_dir: Path,
    evaluation_path: Path | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.csv"
    markdown_path = output_dir / "comparison.md"
    comparison = write_model_comparison(registry_path=registry_path, output_path=comparison_path, evaluation_path=evaluation_path)
    markdown = write_model_markdown_report(comparison_path=comparison, output_path=markdown_path)
    predictions = pd.read_parquet(predictions_path) if predictions_path.suffix == ".parquet" else pd.read_csv(predictions_path)
    review_paths = write_calibration_review(predictions, output_dir / "review")
    paths = {"comparison": comparison, "markdown": markdown}
    paths.update({f"review_{name}": path for name, path in review_paths.items()})
    return paths
