"""Command-line interface for soccer analytics research workflows."""

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from soccer_edge.active_sampling import write_low_confidence_rows
from soccer_edge.annotation_audit import write_annotation_audit
from soccer_edge.annotation_dataset import write_annotation_dataset_config_from_values
from soccer_edge.annotation_split import write_annotation_split
from soccer_edge.annotations import write_detection_annotations_from_table
from soccer_edge.app_logging import configure_logging, get_logger
from soccer_edge.auto_data_card import write_auto_data_card
from soccer_edge.calibration_qa import write_projection_qa_csv, write_projection_qa_svg
from soccer_edge.calibration_summary import write_calibration_summary
from soccer_edge.card_validation import assert_valid_cards
from soccer_edge.cards import write_data_card, write_model_card
from soccer_edge.config import get_settings
from soccer_edge.contact_sheet import write_contact_sheet
from soccer_edge.correction_merge import merge_reviewed_corrections_from_tables
from soccer_edge.correction_review_ui import write_correction_review_assets
from soccer_edge.crop_export import export_image_crops_from_table
from soccer_edge.dataset_versioning import write_dataset_versions
from soccer_edge.evaluation.calibration_review import write_calibration_review
from soccer_edge.evaluation.replay import replay_predictions
from soccer_edge.example_pipeline import run_tiny_example_pipeline
from soccer_edge.features.table_builders import build_inplay_rolling_table, build_prematch_table
from soccer_edge.frame_export import export_video_frame_manifest
from soccer_edge.frame_join import attach_image_paths_from_tables
from soccer_edge.graph_payload_files import write_annotation_audit_payloads, write_graph_payloads
from soccer_edge.ingest.metrica_loader import ingest_metrica as run_metrica_ingest
from soccer_edge.ingest.processed_tables import write_metrica_processed, write_soccernet_processed, write_statsbomb_processed
from soccer_edge.ingest.soccernet_loader import ingest_soccernet as run_soccernet_ingest
from soccer_edge.ingest.statsbomb_loader import ingest_statsbomb as run_statsbomb_ingest
from soccer_edge.ingest.video_discovery import build_candidate
from soccer_edge.local_finetune_pipeline import run_local_finetune_pipeline
from soccer_edge.local_finetune_plan import write_local_finetune_shell_plan
from soccer_edge.local_training_chain import run_local_training_chain
from soccer_edge.media_inference import make_media_callback
from soccer_edge.media_pipeline import run_media_table_stub
from soccer_edge.media_processing import run_media_processing_loop
from soccer_edge.models.bundle import save_bundle
from soccer_edge.models.cnn_predict import export_cnn_bundle_predictions
from soccer_edge.models.cnn_review import write_cnn_calibration_review
from soccer_edge.models.cnn_runner import train_cnn_from_npz
from soccer_edge.models.comparison import write_model_comparison
from soccer_edge.models.markdown_report import write_model_markdown_report
from soccer_edge.models.prediction_export import export_bundle_predictions
from soccer_edge.models.registry import write_registry_index, write_registry_summary
from soccer_edge.models.run_summary import write_run_summary
from soccer_edge.models.simple_classifier import fit_simple_classifier
from soccer_edge.models.tensor_samples import build_npz_from_table
from soccer_edge.object_confusion import write_confusion_outputs
from soccer_edge.object_eval import write_object_eval_metrics
from soccer_edge.object_model import LocalObjectRunner
from soccer_edge.object_training import ObjectTrainingConfig, run_object_training
from soccer_edge.player_stats import write_player_form_features, write_player_match_stats
from soccer_edge.promotion_gate import write_promotion_gate_report
from soccer_edge.raw_data_sources import write_raw_data_sources
from soccer_edge.training_sources import write_training_sources
from soccer_edge.video.batch_runner import build_processing_plan
from soccer_edge.video.calibration_io import load_homography
from soccer_edge.video.local_catalog import write_local_video_catalog
from soccer_edge.video.state_tables import write_video_state_tables

app = typer.Typer(help="Soccer analytics research CLI.")
ingest_app = typer.Typer(help="Ingest open soccer datasets.")
discover_app = typer.Typer(help="Discover candidate video metadata.")
video_app = typer.Typer(help="Process licensed local soccer videos.")
features_app = typer.Typer(help="Build model feature tables.")
train_app = typer.Typer(help="Train probability models.")
model_app = typer.Typer(help="Save, score, and inspect model outputs.")
examples_app = typer.Typer(help="Run tiny local examples.")

app.add_typer(ingest_app, name="ingest")
app.add_typer(discover_app, name="discover")
app.add_typer(video_app, name="video")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(model_app, name="model")
app.add_typer(examples_app, name="examples")

console = Console()
logger = get_logger("soccer_edge.cli")


def parse_paths(value: str | None) -> list[Path] | None:
    if value is None:
        return None
    return [Path(item.strip()) for item in value.split(",") if item.strip()]


def parse_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


@app.callback()
def main(json_logs: bool = typer.Option(False, help="Emit structured JSON logs.")) -> None:
    """Configure global CLI options."""

    configure_logging(json_logs=json_logs)
    logger.debug("cli configured")


@app.command()
def doctor() -> None:
    """Print environment defaults."""

    settings = get_settings()
    console.print("Soccer analytics environment", style="bold")
    console.print(f"env={settings.env}")
    console.print(f"data_dir={settings.data_dir}")
    console.print(f"external_execution_enabled={settings.external_execution_enabled}")


@ingest_app.command("statsbomb")
def ingest_statsbomb(path: Path = typer.Option(..., exists=False)) -> None:
    """Prepare StatsBomb ingest."""

    console.print(run_statsbomb_ingest(path))


@ingest_app.command("metrica")
def ingest_metrica(path: Path = typer.Option(..., exists=False)) -> None:
    """Prepare Metrica ingest."""

    console.print(run_metrica_ingest(path))


@ingest_app.command("soccernet")
def ingest_soccernet(path: Path = typer.Option(..., exists=False)) -> None:
    """Prepare SoccerNet ingest."""

    console.print(run_soccernet_ingest(path))


@ingest_app.command("raw-sources")
def ingest_raw_sources(output: Path = typer.Option(Path("data/processed/raw_data_sources.csv"))) -> None:
    """Write a rights-aware catalog of candidate raw data sources."""

    path = write_raw_data_sources(output)
    console.print(f"wrote={path}")


@ingest_app.command("write-processed")
def ingest_write_processed(
    source: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("data/processed/ingest")),
    source_type: str = typer.Option(..., help="statsbomb, metrica, or soccernet"),
    dataset_version: str = typer.Option("unknown"),
) -> None:
    """Write local source files into processed parquet tables with lineage."""

    if source_type == "statsbomb":
        paths = write_statsbomb_processed(source, output_dir, dataset_version)
    elif source_type == "metrica":
        paths = write_metrica_processed(source, output_dir, dataset_version)
    elif source_type == "soccernet":
        paths = write_soccernet_processed(source, output_dir, dataset_version)
    else:
        raise typer.BadParameter("source_type must be statsbomb, metrica, or soccernet")
    console.print({name: str(path) for name, path in paths.items()})


@discover_app.command("video")
def discover_video(
    query: str = typer.Option(...),
    url: str = typer.Option("https://example.com/manual-review"),
    title: str = typer.Option("Manual review candidate"),
) -> None:
    """Store candidate video metadata only."""

    candidate = build_candidate(url=url, title=title, query=query)
    console.print(candidate)


@video_app.command("catalog-local")
def catalog_local_video(
    root: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("manifests/local_video_manifest.csv")),
    rights_status: str = typer.Option("owned"),
    clip_type: str = typer.Option("full_match"),
) -> None:
    """Catalog approved local footage into a manifest."""

    path = write_local_video_catalog(root=root, output=output, rights_status=rights_status, clip_type=clip_type)
    console.print(f"wrote={path}")


@video_app.command("plan")
def plan_video_processing(
    manifest: Path = typer.Option(..., exists=True),
    licensed_root: Path = typer.Option(Path("data/raw/video_licensed")),
) -> None:
    """Plan processable local video rows from a manifest."""

    plan = build_processing_plan(manifest, licensed_root)
    console.print(f"processable={len(plan.processable)} skipped={len(plan.skipped)}")


@video_app.command("process")
def process_video(
    input_path: Path = typer.Option(..., "--input", exists=True),
    output_dir: Path = typer.Option(Path("data/processed/video_pipeline")),
    frame_count: int = typer.Option(0),
) -> None:
    """Run the first local licensed video processing stub."""

    result = run_media_table_stub(input_path=input_path, output_dir=output_dir, frame_count=frame_count)
    console.print(result)


@video_app.command("export-frames")
def export_frames(
    input_path: Path = typer.Option(..., "--input", exists=True),
    output_dir: Path = typer.Option(Path("data/processed/frames")),
    manifest_output: Path = typer.Option(Path("data/processed/frame_manifest.csv")),
    stride: int = typer.Option(1),
    max_frames: int | None = typer.Option(None),
) -> None:
    """Export local video frames and a manifest with image paths."""

    path = export_video_frame_manifest(input_path, output_dir, manifest_output, stride=stride, max_frames=max_frames)
    console.print(f"wrote={path}")


@video_app.command("process-local-model")
def process_video_local_model(
    input_path: Path = typer.Option(..., "--input", exists=True),
    model_path: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("data/processed/video_model")),
    stride: int = typer.Option(1),
    max_samples: int | None = typer.Option(None),
    calibration: Path | None = typer.Option(None, exists=False),
) -> None:
    """Run optional local object-model inference over approved local footage."""

    runner = LocalObjectRunner(model_path)
    callback = make_media_callback(runner)
    transform = load_homography(calibration) if calibration is not None else None
    paths = run_media_processing_loop(input_path=input_path, output_dir=output_dir, callback=callback, stride=stride, max_samples=max_samples, transform=transform)
    console.print({name: str(path) for name, path in paths.items()})


@video_app.command("attach-frame-images")
def attach_frame_images(
    detections: Path = typer.Option(..., exists=True),
    frame_manifest: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/detections_with_images.csv")),
    frame_column: str = typer.Option("frame_idx"),
    image_path_column: str = typer.Option("image_path"),
) -> None:
    """Attach exported frame image paths to detection rows by frame index."""

    path = attach_image_paths_from_tables(detections, frame_manifest, output, frame_column=frame_column, image_path_column=image_path_column)
    console.print(f"wrote={path}")


@video_app.command("audit-annotations")
def audit_annotations(
    source: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("data/processed/annotation_audit")),
    class_column: str = typer.Option("class_name"),
    frame_column: str = typer.Option("frame_idx"),
    split_column: str = typer.Option("split"),
) -> None:
    """Write annotation label audit summaries by class, frame, and split."""

    paths = write_annotation_audit(source, output_dir, class_column=class_column, frame_column=frame_column, split_column=split_column)
    console.print({name: str(path) for name, path in paths.items()})


@video_app.command("dataset-versions")
def dataset_versions_command(
    paths: str = typer.Option(..., help="Comma-separated local file paths to version."),
    output: Path = typer.Option(Path("data/processed/dataset_versions.csv")),
) -> None:
    """Write deterministic file hashes and table shapes for dataset assets."""

    selected = parse_paths(paths) or []
    path = write_dataset_versions(selected, output)
    console.print(f"wrote={path}")


@video_app.command("correction-review")
def correction_review(
    source: Path = typer.Option(..., exists=True),
    html_output: Path = typer.Option(Path("data/processed/correction_review.html")),
    template_output: Path = typer.Option(Path("data/processed/reviewed_corrections.csv")),
    keys: str = typer.Option("crop_path"),
    image_column: str = typer.Option("crop_path"),
    class_column: str = typer.Option("class_name"),
) -> None:
    """Write a correction-review HTML page and CSV template."""

    key_columns = [column.strip() for column in keys.split(",") if column.strip()]
    paths = write_correction_review_assets(source, html_output, template_output, key_columns, image_column=image_column, class_column=class_column)
    console.print({name: str(path) for name, path in paths.items()})


@video_app.command("merge-corrections")
def merge_corrections(
    base: Path = typer.Option(..., exists=True),
    corrections: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/corrected_detections.csv")),
    keys: str = typer.Option("crop_path"),
    action_column: str = typer.Option("review_action"),
) -> None:
    """Merge reviewed low-confidence crop corrections back into detection rows."""

    key_columns = [column.strip() for column in keys.split(",") if column.strip()]
    path = merge_reviewed_corrections_from_tables(base, corrections, output, key_columns=key_columns, action_column=action_column)
    console.print(f"wrote={path}")


@video_app.command("export-annotations")
def export_annotations(
    source: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("data/processed/annotations")),
    classes: str = typer.Option("player,ball"),
    image_width: float = typer.Option(1920.0),
    image_height: float = typer.Option(1080.0),
) -> None:
    """Export normalized box annotations from detection rows."""

    class_names = [class_name.strip() for class_name in classes.split(",") if class_name.strip()]
    paths = write_detection_annotations_from_table(source, output_dir, class_names, image_width, image_height)
    console.print({name: str(path) for name, path in paths.items()})


@video_app.command("split-annotations")
def split_annotations(
    source: Path = typer.Option(..., exists=True),
    train_output: Path = typer.Option(Path("data/processed/annotations/train.csv")),
    val_output: Path = typer.Option(Path("data/processed/annotations/val.csv")),
    train_fraction: float = typer.Option(0.8),
    group_column: str = typer.Option("frame_idx"),
) -> None:
    """Split annotation rows into train and validation CSVs."""

    paths = write_annotation_split(source, train_output, val_output, train_fraction=train_fraction, group_column=group_column)
    console.print({name: str(path) for name, path in paths.items()})


@video_app.command("annotation-config")
def annotation_config(
    root: Path = typer.Option(...),
    train_images: Path = typer.Option(...),
    val_images: Path = typer.Option(...),
    classes: str = typer.Option("player,ball"),
    output: Path = typer.Option(Path("data/processed/annotations/data.yaml")),
    test_images: Path | None = typer.Option(None),
) -> None:
    """Write annotation dataset config for local object-model training."""

    class_names = [class_name.strip() for class_name in classes.split(",") if class_name.strip()]
    path = write_annotation_dataset_config_from_values(root, train_images, val_images, class_names, output, test_images=test_images)
    console.print(f"wrote={path}")


@video_app.command("sample-low-confidence")
def sample_low_confidence(
    source: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/low_confidence.csv")),
    threshold: float = typer.Option(0.5),
    limit: int | None = typer.Option(None),
) -> None:
    """Write low-confidence detection rows for review."""

    path = write_low_confidence_rows(source, output, threshold=threshold, limit=limit)
    console.print(f"wrote={path}")


@video_app.command("export-crops")
def export_crops(
    source: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("data/processed/crops")),
    manifest_output: Path = typer.Option(Path("data/processed/crop_manifest.csv")),
    image_path_column: str = typer.Option("image_path"),
) -> None:
    """Export object crops from local frame image rows."""

    path = export_image_crops_from_table(source, output_dir, manifest_output, image_path_column=image_path_column)
    console.print(f"wrote={path}")


@video_app.command("contact-sheet")
def crop_contact_sheet(
    source: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/crop_review.html")),
    title: str = typer.Option("Crop Review"),
    image_column: str = typer.Option("crop_path"),
) -> None:
    """Write an HTML contact sheet for crop review."""

    path = write_contact_sheet(source, output, title=title, image_column=image_column)
    console.print(f"wrote={path}")


@video_app.command("calibration-qa")
def calibration_qa(
    calibration: Path = typer.Option(..., exists=True),
    csv_output: Path = typer.Option(Path("data/processed/calibration_qa.csv")),
    svg_output: Path = typer.Option(Path("data/processed/calibration_qa.svg")),
) -> None:
    """Write calibration error table and SVG visual QA."""

    csv_path = write_projection_qa_csv(calibration, csv_output)
    svg_path = write_projection_qa_svg(calibration, svg_output)
    console.print({"csv": str(csv_path), "svg": str(svg_path)})


@video_app.command("calibration-summary")
def calibration_summary(
    source: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/calibration_qa.md")),
    title: str = typer.Option("Calibration QA Summary"),
) -> None:
    """Write calibration error summary markdown from QA CSV."""

    path = write_calibration_summary(source, output, title=title)
    console.print(f"wrote={path}")


@features_app.command("build")
def build_features(output_dir: Path = typer.Option(Path("data/processed/state_tables"))) -> None:
    """Create empty state-table files as the first feature-build target."""

    paths = write_video_state_tables(output_dir)
    console.print({name: str(path) for name, path in paths.items()})


@features_app.command("prematch")
def build_prematch_features(
    matches: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/prematch_features.parquet")),
) -> None:
    """Build a prematch feature table from match rows."""

    frame = pd.read_parquet(matches) if matches.suffix == ".parquet" else pd.read_csv(matches)
    table = build_prematch_table(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    console.print(f"wrote={output} rows={len(table)}")


@features_app.command("inplay")
def build_inplay_features(
    source: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/inplay_features.parquet")),
    columns: str = typer.Option(..., help="Comma-separated feature columns."),
    window_seconds: float = typer.Option(60.0),
) -> None:
    """Build rolling in-play feature rows."""

    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    feature_columns = [column.strip() for column in columns.split(",") if column.strip()]
    table = build_inplay_rolling_table(frame, feature_columns=feature_columns, window_seconds=window_seconds)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output, index=False)
    console.print(f"wrote={output} rows={len(table)}")


@features_app.command("player-stats")
def build_player_stats(
    events: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/player_match_stats.csv")),
) -> None:
    """Build player-match stats from event rows."""

    path = write_player_match_stats(events, output)
    console.print(f"wrote={path}")


@features_app.command("player-form")
def build_player_form(
    player_stats: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/player_form.csv")),
    window: int = typer.Option(5),
    order_column: str = typer.Option("match_id"),
) -> None:
    """Build leakage-safe rolling player form features."""

    path = write_player_form_features(player_stats, output, window=window, order_column=order_column)
    console.print(f"wrote={path}")


@features_app.command("tensor-samples")
def build_tensor_samples(
    source: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/tensor_samples.npz")),
    columns: str = typer.Option(..., help="Comma-separated flattened spatial grid columns."),
    label: str = typer.Option("label"),
    sequence_length: int = typer.Option(1),
    channels: int = typer.Option(3),
    height: int = typer.Option(8),
    width: int = typer.Option(8),
    group: str | None = typer.Option(None, help="Optional group column such as match_id."),
    order: str | None = typer.Option(None, help="Optional ordering column such as timestamp_seconds."),
) -> None:
    """Build an NPZ tensor dataset for CNN training."""

    spatial_columns = [column.strip() for column in columns.split(",") if column.strip()]
    path = build_npz_from_table(
        source=source,
        output_path=output,
        spatial_columns=spatial_columns,
        label_column=label,
        sequence_length=sequence_length,
        channels=channels,
        height=height,
        width=width,
        group_column=group,
        order_column=order,
    )
    console.print(f"wrote={path}")


@train_app.command("simple")
def train_simple(
    source: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("data/processed/simple_model")),
    columns: str = typer.Option(..., help="Comma-separated feature columns."),
    label: str = typer.Option("label"),
) -> None:
    """Fit a simple sklearn classifier from a table."""

    frame = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    feature_columns = [column.strip() for column in columns.split(",") if column.strip()]
    paths = fit_simple_classifier(frame, feature_columns=feature_columns, label_column=label, output_dir=output_dir)
    console.print({name: str(path) for name, path in paths.items()})


@train_app.command("cnn")
def train_cnn(
    source: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("data/processed/cnn_model")),
    output_classes: int = typer.Option(3),
    epochs: int = typer.Option(1),
    batch_size: int = typer.Option(4),
    hidden_size: int = typer.Option(128),
) -> None:
    """Fit a CNN from an NPZ tensor dataset."""

    paths = train_cnn_from_npz(
        source,
        output_dir,
        output_classes=output_classes,
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
    )
    console.print({name: str(path) for name, path in paths.items()})


@train_app.command("local-chain")
def train_local_chain(
    footage_root: Path = typer.Option(..., exists=True),
    tabular_source: Path = typer.Option(..., exists=True),
    grid_source: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("data/processed/local_training_chain")),
    tabular_columns: str = typer.Option(...),
    grid_columns: str = typer.Option(...),
    label: str = typer.Option("label"),
    rights_status: str = typer.Option("owned"),
    group: str | None = typer.Option("match_id"),
    order: str | None = typer.Option(None),
    detection_source: Path | None = typer.Option(None, exists=False),
) -> None:
    """Run the local chain from footage catalog through review artifacts."""

    paths = run_local_training_chain(
        footage_root=footage_root,
        output_dir=output_dir,
        tabular_source=tabular_source,
        grid_source=grid_source,
        tabular_columns=[column.strip() for column in tabular_columns.split(",") if column.strip()],
        grid_columns=[column.strip() for column in grid_columns.split(",") if column.strip()],
        label_column=label,
        rights_status=rights_status,
        group_column=group,
        order_column=order,
        detection_source=detection_source,
    )
    console.print({name: str(path) for name, path in paths.items()})


@train_app.command("local-finetune")
def train_local_finetune(
    input_path: Path = typer.Option(..., "--input", exists=False),
    object_model_path: Path = typer.Option(..., exists=False),
    output_dir: Path = typer.Option(Path("data/processed/local_finetune")),
    classes: str = typer.Option("player,ball"),
    base_model_path: Path | None = typer.Option(None, exists=False),
    calibration_path: Path | None = typer.Option(None, exists=False),
    stride: int = typer.Option(5),
    max_frames: int | None = typer.Option(100),
    train_fraction: float = typer.Option(0.8),
    threshold: float = typer.Option(0.5),
    train_object_model: bool = typer.Option(False),
    dry_run_plan: Path | None = typer.Option(None, exists=False),
    validate_plan_inputs: bool = typer.Option(False),
) -> None:
    """Run the local fine-tuning path from frames through optional object training."""

    if dry_run_plan is not None:
        plan_path = write_local_finetune_shell_plan(
            output_path=dry_run_plan,
            input_path=input_path,
            object_model_path=object_model_path,
            output_dir=output_dir,
            classes=classes,
            calibration_path=calibration_path,
            stride=stride,
            max_frames=max_frames,
            threshold=threshold,
            train_fraction=train_fraction,
            validate_inputs=validate_plan_inputs,
        )
        console.print(f"wrote={plan_path}")
        return
    if not input_path.exists() or not object_model_path.exists():
        raise typer.BadParameter("--input and --object-model-path must exist unless --dry-run-plan is used")
    class_names = [class_name.strip() for class_name in classes.split(",") if class_name.strip()]
    outputs = run_local_finetune_pipeline(
        input_path=input_path,
        object_model_path=object_model_path,
        output_dir=output_dir,
        classes=class_names,
        base_model_path=base_model_path,
        calibration_path=calibration_path,
        stride=stride,
        max_frames=max_frames,
        train_fraction=train_fraction,
        threshold=threshold,
        train_object_model=train_object_model,
    )
    console.print({name: str(path) if path is not None else None for name, path in outputs.__dict__.items()})


@train_app.command("object-model")
def train_object_model(
    data_config: Path = typer.Option(..., exists=True),
    base_model: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("data/processed/object_training")),
    run_name: str = typer.Option("local_object_model"),
    epochs: int = typer.Option(50),
    image_size: int = typer.Option(640),
) -> None:
    """Run optional local object-model training."""

    config = ObjectTrainingConfig(data_config=data_config, base_model=base_model, output_dir=output_dir, run_name=run_name, epochs=epochs, image_size=image_size)
    paths = run_object_training(config)
    console.print({name: str(path) for name, path in paths.items()})


@train_app.command("prematch")
def train_prematch() -> None:
    """Train prematch model placeholder."""

    console.print("Use: soccer-edge train simple --source <table> --columns <features> --label <label>")


@train_app.command("inplay")
def train_inplay() -> None:
    """Train in-play model placeholder."""

    console.print("Use: soccer-edge train simple --source <table> --columns <features> --label <label>")


@model_app.command("save-demo")
def save_demo_model(output_dir: Path = typer.Option(Path("data/processed/model_demo"))) -> None:
    """Save a small model bundle with metadata."""

    paths = save_bundle(
        model={"kind": "demo"},
        output_dir=output_dir,
        name="demo",
        version="v0",
        feature_names=["demo_feature"],
        metrics={"accuracy": 0.0},
        notes="Demo bundle for pipeline validation.",
    )
    console.print({name: str(path) for name, path in paths.items()})


@model_app.command("registry")
def model_registry(
    root_dir: Path = typer.Option(Path("data/processed")),
    output: Path = typer.Option(Path("data/processed/model_registry.csv")),
) -> None:
    """Build an index of saved model bundles."""

    path = write_registry_index(root_dir, output)
    console.print(f"wrote={path}")


@model_app.command("registry-summary")
def model_registry_summary(
    root_dir: Path = typer.Option(Path("data/processed")),
    output: Path = typer.Option(Path("data/processed/model_registry_summary.csv")),
    metric: str = typer.Option("accuracy"),
) -> None:
    """Build a sorted summary of saved model bundles."""

    path = write_registry_summary(root_dir, output, metric=metric)
    console.print(f"wrote={path}")


@model_app.command("predict")
def model_predict(
    bundle_dir: Path = typer.Option(..., exists=True),
    source: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/predictions.csv")),
    columns: str | None = typer.Option(None, help="Optional comma-separated feature override."),
) -> None:
    """Write predictions from a saved model bundle."""

    feature_columns = None if columns is None else [column.strip() for column in columns.split(",") if column.strip()]
    path = export_bundle_predictions(bundle_dir=bundle_dir, source=source, output=output, feature_columns=feature_columns)
    console.print(f"wrote={path}")


@model_app.command("predict-cnn")
def model_predict_cnn(
    bundle_dir: Path = typer.Option(..., exists=True),
    source: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/cnn_predictions.csv")),
    batch_size: int = typer.Option(8),
) -> None:
    """Write CNN predictions from a saved tensor model bundle."""

    path = export_cnn_bundle_predictions(bundle_dir=bundle_dir, npz_path=source, output=output, batch_size=batch_size)
    console.print(f"wrote={path}")


@model_app.command("compare")
def model_compare(
    registry: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/model_comparison.csv")),
    evaluation: Path | None = typer.Option(None, exists=False),
) -> None:
    """Write a model comparison report."""

    path = write_model_comparison(registry_path=registry, output_path=output, evaluation_path=evaluation)
    console.print(f"wrote={path}")


@model_app.command("compare-markdown")
def model_compare_markdown(
    comparison: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/model_comparison.md")),
) -> None:
    """Write a markdown model comparison report."""

    path = write_model_markdown_report(comparison_path=comparison, output_path=output)
    console.print(f"wrote={path}")


@model_app.command("run-summary")
def model_run_summary(
    registry: Path = typer.Option(..., exists=True),
    predictions: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("data/processed/run_summary")),
    evaluation: Path | None = typer.Option(None, exists=False),
) -> None:
    """Write comparison, markdown, and calibration artifacts together."""

    paths = write_run_summary(registry_path=registry, predictions_path=predictions, output_dir=output_dir, evaluation_path=evaluation)
    console.print({name: str(path) for name, path in paths.items()})


@model_app.command("model-card")
def model_card(
    bundle_dir: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("MODEL_CARD.md")),
    dataset_id: str | None = typer.Option(None),
    version_paths: str | None = typer.Option(None, help="Optional comma-separated dataset paths to hash."),
    graph_ids: str | None = typer.Option(None, help="Optional comma-separated graph payload IDs."),
) -> None:
    """Write a model card for a saved bundle."""

    path = write_model_card(bundle_dir, output, dataset_id=dataset_id, version_paths=parse_paths(version_paths), graph_ids=parse_strings(graph_ids))
    console.print(f"wrote={path}")


@model_app.command("data-card")
def data_card(
    dataset_name: str = typer.Option(...),
    sources: str = typer.Option(..., help="Comma-separated source paths."),
    output: Path = typer.Option(Path("DATA_CARD.md")),
    rights_status: str = typer.Option("owned"),
    dataset_id: str | None = typer.Option(None),
    version_paths: str | None = typer.Option(None, help="Optional comma-separated dataset paths to hash."),
    graph_ids: str | None = typer.Option(None, help="Optional comma-separated graph payload IDs."),
) -> None:
    """Write a data card for approved local/open sources."""

    source_paths = parse_paths(sources) or []
    path = write_data_card(dataset_name, source_paths, output, rights_status=rights_status, dataset_id=dataset_id, version_paths=parse_paths(version_paths), graph_ids=parse_strings(graph_ids))
    console.print(f"wrote={path}")


@model_app.command("auto-data-card")
def auto_data_card(
    dataset_name: str = typer.Option(...),
    manifests: str = typer.Option(..., help="Comma-separated manifest/table paths."),
    output: Path = typer.Option(Path("DATA_CARD.md")),
    rights_status: str = typer.Option("owned"),
    version_paths: str | None = typer.Option(None, help="Optional comma-separated paths to hash."),
) -> None:
    """Write a data card populated from source catalog and manifest stats."""

    manifest_paths = parse_paths(manifests) or []
    path = write_auto_data_card(dataset_name, manifest_paths, output, rights_status=rights_status, version_paths=parse_paths(version_paths))
    console.print(f"wrote={path}")


@model_app.command("source-catalog")
def source_catalog(output: Path = typer.Option(Path("data/processed/training_sources.csv"))) -> None:
    """Write the default rights-aware training source catalog."""

    path = write_training_sources(output)
    console.print(f"wrote={path}")


@model_app.command("object-eval")
def object_eval(
    source: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/object_eval.csv")),
    class_column: str = typer.Option("class_name"),
    status_column: str = typer.Option("status"),
) -> None:
    """Write object-model precision, recall, and F1 by class."""

    path = write_object_eval_metrics(source, output, class_column=class_column, status_column=status_column)
    console.print(f"wrote={path}")


@model_app.command("object-confusion")
def object_confusion(
    source: Path = typer.Option(..., exists=True),
    table_output: Path = typer.Option(Path("data/processed/object_confusion.csv")),
    svg_output: Path = typer.Option(Path("data/processed/object_confusion.svg")),
    actual_column: str = typer.Option("actual_class"),
    predicted_column: str = typer.Option("predicted_class"),
) -> None:
    """Write object-model confusion matrix table and SVG."""

    paths = write_confusion_outputs(source, table_output, svg_output, actual_column=actual_column, predicted_column=predicted_column)
    console.print({name: str(path) for name, path in paths.items()})


@model_app.command("graph-payloads")
def graph_payloads(
    source: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/graph_payloads.jsonl")),
    kind: str = typer.Option(..., help="dataset-version, annotation-audit, or object-evaluation"),
) -> None:
    """Write graph payload JSONL for one source table."""

    path = write_graph_payloads(source, output, kind)
    console.print(f"wrote={path}")


@model_app.command("graph-audit-payloads")
def graph_audit_payloads(
    audit_dir: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/annotation_audit_payloads.jsonl")),
) -> None:
    """Write graph payload JSONL for all annotation audit CSVs."""

    path = write_annotation_audit_payloads(audit_dir, output)
    console.print(f"wrote={path}")


@model_app.command("promotion-gate")
def promotion_gate(
    model_card_path: Path | None = typer.Option(None, exists=False),
    data_card_path: Path | None = typer.Option(None, exists=False),
    dataset_versions: Path = typer.Option(..., exists=True),
    audit_dir: Path = typer.Option(..., exists=True),
    object_metrics: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(Path("data/processed/promotion_gate.md")),
    min_f1: float = typer.Option(0.0),
) -> None:
    """Validate cards, versions, audits, and object metrics before promotion."""

    path = write_promotion_gate_report(output, model_card_path, data_card_path, dataset_versions, audit_dir, object_metrics, min_f1=min_f1)
    console.print(f"wrote={path}")


@model_app.command("validate-cards")
def validate_cards(
    model_card_path: Path | None = typer.Option(None, exists=False),
    data_card_path: Path | None = typer.Option(None, exists=False),
) -> None:
    """Validate model and data card required sections."""

    assert_valid_cards(model_card_path, data_card_path)
    console.print("cards_valid=true")


@model_app.command("calibration-review-cnn")
def calibration_review_cnn(
    bundle_dir: Path = typer.Option(..., exists=True),
    source: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("data/processed/cnn_calibration_review")),
    batch_size: int = typer.Option(8),
    num_bins: int = typer.Option(10),
) -> None:
    """Write CNN metrics and calibration artifacts from tensor samples."""

    paths = write_cnn_calibration_review(bundle_dir, source, output_dir, batch_size=batch_size, num_bins=num_bins)
    console.print({name: str(path) for name, path in paths.items()})


@model_app.command("evaluate")
def evaluate_model(predictions: Path = typer.Option(..., exists=True)) -> None:
    """Evaluate a CSV or parquet file with label/probability columns."""

    frame = pd.read_parquet(predictions) if predictions.suffix == ".parquet" else pd.read_csv(predictions)
    result = replay_predictions(frame)
    console.print(result)


@model_app.command("calibration-review")
def calibration_review(
    predictions: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(Path("data/processed/calibration_review")),
    num_bins: int = typer.Option(10),
) -> None:
    """Write metrics and calibration artifacts from probability rows."""

    frame = pd.read_parquet(predictions) if predictions.suffix == ".parquet" else pd.read_csv(predictions)
    paths = write_calibration_review(frame, output_dir=output_dir, num_bins=num_bins)
    console.print({name: str(path) for name, path in paths.items()})


@examples_app.command("tiny")
def examples_tiny(
    repo_root: Path = typer.Option(Path(".")),
    output_dir: Path = typer.Option(Path("data/processed/examples/tiny_pipeline")),
) -> None:
    """Run the tiny local example pipeline end to end."""

    paths = run_tiny_example_pipeline(repo_root=repo_root, output_dir=output_dir)
    console.print({name: str(path) for name, path in paths.items()})


@app.command()
def calibrate() -> None:
    """Calibrate model probabilities."""

    console.print("Use: soccer-edge model calibration-review --predictions <csv-or-parquet>")


@app.command()
def evaluate() -> None:
    """Evaluate models offline."""

    console.print("Use: soccer-edge model evaluate --predictions <csv-or-parquet>")
