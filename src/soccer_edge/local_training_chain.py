from pathlib import Path

import pandas as pd

from soccer_edge.active_sampling import write_low_confidence_rows
from soccer_edge.annotations import write_detection_annotations_from_table
from soccer_edge.cards import write_data_card, write_model_card
from soccer_edge.models.prediction_export import export_bundle_predictions
from soccer_edge.models.registry import write_registry_index, write_registry_summary
from soccer_edge.models.run_summary import write_run_summary
from soccer_edge.models.simple_classifier import fit_simple_classifier
from soccer_edge.models.tensor_samples import build_npz_from_table
from soccer_edge.video.local_catalog import write_local_video_catalog


def run_local_training_chain(
    footage_root: Path,
    output_dir: Path,
    tabular_source: Path,
    grid_source: Path,
    tabular_columns: list[str],
    grid_columns: list[str],
    label_column: str = "label",
    rights_status: str = "owned",
    group_column: str | None = "match_id",
    order_column: str | None = None,
    detection_source: Path | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_local_video_catalog(footage_root, output_dir / "local_video_manifest.csv", rights_status=rights_status)
    model_dir = output_dir / "simple_model"
    predictions_path = output_dir / "predictions.csv"
    registry_path = output_dir / "registry.csv"
    summary_path = output_dir / "registry_summary.csv"
    tensor_path = output_dir / "tensor_samples.npz"

    fit_simple_classifier(pd.read_csv(tabular_source), tabular_columns, label_column, model_dir)
    export_bundle_predictions(model_dir, tabular_source, predictions_path)
    write_registry_index(output_dir, registry_path)
    write_registry_summary(output_dir, summary_path)
    review_paths = write_run_summary(summary_path, predictions_path, output_dir / "run_summary")
    build_npz_from_table(
        grid_source,
        tensor_path,
        grid_columns,
        label_column,
        sequence_length=2,
        channels=1,
        height=2,
        width=2,
        group_column=group_column,
        order_column=order_column,
    )
    model_card = write_model_card(model_dir, output_dir / "MODEL_CARD.md")
    data_card = write_data_card("local-training-chain", [footage_root, tabular_source, grid_source], output_dir / "DATA_CARD.md", rights_status=rights_status)

    paths = {
        "manifest": manifest_path,
        "model": model_dir / "model.joblib",
        "metadata": model_dir / "metadata.json",
        "predictions": predictions_path,
        "registry": registry_path,
        "summary": summary_path,
        "tensor_samples": tensor_path,
        "model_card": model_card,
        "data_card": data_card,
    }
    paths.update(review_paths)

    if detection_source is not None:
        annotations = write_detection_annotations_from_table(detection_source, output_dir / "annotations", ["player", "ball"], 1920, 1080)
        low_confidence = write_low_confidence_rows(detection_source, output_dir / "low_confidence.csv")
        paths["annotations_classes"] = annotations["classes"]
        paths["low_confidence"] = low_confidence
    return paths
