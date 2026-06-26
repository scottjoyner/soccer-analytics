from pathlib import Path

import pandas as pd

from soccer_edge.models.comparison import write_model_comparison
from soccer_edge.models.markdown_report import write_model_markdown_report
from soccer_edge.models.prediction_export import export_bundle_predictions
from soccer_edge.models.registry import write_registry_index, write_registry_summary
from soccer_edge.models.simple_classifier import fit_simple_classifier
from soccer_edge.models.tensor_samples import build_npz_from_table


def run_tiny_example_pipeline(repo_root: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    training = repo_root / "examples" / "tiny_training.csv"
    grid = repo_root / "examples" / "tiny_grid_features.csv"
    model_dir = output_dir / "simple_model"
    tensor_path = output_dir / "tiny_tensor_samples.npz"
    predictions_path = output_dir / "predictions.csv"
    registry_path = output_dir / "registry.csv"
    summary_path = output_dir / "registry_summary.csv"
    comparison_path = output_dir / "comparison.csv"
    markdown_path = output_dir / "comparison.md"

    training_frame = pd.read_csv(training)
    fit_simple_classifier(training_frame, ["speed_last", "pressure_last"], "label", model_dir)
    export_bundle_predictions(model_dir, training, predictions_path)
    write_registry_index(output_dir, registry_path)
    write_registry_summary(output_dir, summary_path)
    write_model_comparison(summary_path, comparison_path)
    write_model_markdown_report(comparison_path, markdown_path)
    build_npz_from_table(
        grid,
        tensor_path,
        ["g0", "g1", "g2", "g3"],
        "label",
        sequence_length=2,
        channels=1,
        height=2,
        width=2,
        group_column="match_id",
    )
    return {
        "model": model_dir / "model.joblib",
        "metadata": model_dir / "metadata.json",
        "predictions": predictions_path,
        "registry": registry_path,
        "summary": summary_path,
        "comparison": comparison_path,
        "markdown": markdown_path,
        "tensor_samples": tensor_path,
    }
