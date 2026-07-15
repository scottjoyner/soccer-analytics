from dataclasses import dataclass
from pathlib import Path

from soccer_edge.active_sampling import write_low_confidence_rows
from soccer_edge.annotation_split import write_annotation_split
from soccer_edge.annotations import arrange_yolo_dataset_from_table, write_detection_annotations_from_table
from soccer_edge.auto_data_card import write_auto_data_card
from soccer_edge.contact_sheet import write_contact_sheet
from soccer_edge.crop_export import export_image_crops_from_table
from soccer_edge.frame_export import export_video_frame_manifest
from soccer_edge.frame_join import attach_image_paths_from_tables
from soccer_edge.media_inference import make_media_callback
from soccer_edge.media_processing import run_media_processing_loop
from soccer_edge.object_model import LocalObjectRunner
from soccer_edge.object_training import ObjectTrainingConfig, run_object_training
from soccer_edge.video.calibration_io import load_homography


@dataclass(frozen=True)
class LocalFinetuneOutputs:
    frame_manifest: Path
    detections: Path
    detections_with_images: Path
    low_confidence: Path
    crop_manifest: Path
    contact_sheet: Path
    annotations_dir: Path
    train_annotations: Path
    val_annotations: Path
    object_dataset_dir: Path
    annotation_config: Path
    data_card: Path
    object_run_dir: Path | None = None


def local_finetune_outputs(output_dir: Path) -> LocalFinetuneOutputs:
    annotations_dir = output_dir / "annotations"
    object_dataset_dir = annotations_dir / "yolo"
    return LocalFinetuneOutputs(
        frame_manifest=output_dir / "frame_manifest.csv",
        detections=output_dir / "video_model" / "detections.parquet",
        detections_with_images=output_dir / "detections_with_images.csv",
        low_confidence=output_dir / "low_confidence.csv",
        crop_manifest=output_dir / "crop_manifest.csv",
        contact_sheet=output_dir / "crop_review.html",
        annotations_dir=annotations_dir,
        train_annotations=annotations_dir / "train.csv",
        val_annotations=annotations_dir / "val.csv",
        object_dataset_dir=object_dataset_dir,
        annotation_config=object_dataset_dir / "data.yaml",
        data_card=output_dir / "DATA_CARD.md",
        object_run_dir=output_dir / "object_training" / "local_object_model",
    )


def run_local_finetune_pipeline(
    input_path: Path,
    object_model_path: Path,
    output_dir: Path,
    classes: list[str],
    base_model_path: Path | None = None,
    calibration_path: Path | None = None,
    stride: int = 5,
    max_frames: int | None = 100,
    train_fraction: float = 0.8,
    threshold: float = 0.5,
    image_width: float = 1920.0,
    image_height: float = 1080.0,
    train_object_model: bool = False,
    object_epochs: int = 50,
    object_image_size: int = 640,
) -> LocalFinetuneOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = local_finetune_outputs(output_dir)
    frames_dir = output_dir / "frames"
    crops_dir = output_dir / "crops"

    export_video_frame_manifest(input_path, frames_dir, outputs.frame_manifest, stride=stride, max_frames=max_frames)

    runner = LocalObjectRunner(object_model_path)
    callback = make_media_callback(runner)
    transform = load_homography(calibration_path) if calibration_path is not None else None
    run_media_processing_loop(
        input_path=input_path,
        output_dir=output_dir / "video_model",
        callback=callback,
        stride=stride,
        max_samples=max_frames,
        transform=transform,
    )

    attach_image_paths_from_tables(outputs.detections, outputs.frame_manifest, outputs.detections_with_images)
    write_low_confidence_rows(outputs.detections_with_images, outputs.low_confidence, threshold=threshold)
    export_image_crops_from_table(outputs.low_confidence, crops_dir, outputs.crop_manifest)
    write_contact_sheet(outputs.crop_manifest, outputs.contact_sheet)
    write_detection_annotations_from_table(outputs.detections_with_images, outputs.annotations_dir, classes, image_width, image_height)
    write_annotation_split(outputs.detections_with_images, outputs.train_annotations, outputs.val_annotations, train_fraction=train_fraction)
    arrange_yolo_dataset_from_table(
        outputs.detections_with_images,
        outputs.object_dataset_dir,
        classes,
        image_width,
        image_height,
        train_fraction=train_fraction,
        image_column="image_path",
    )
    write_auto_data_card(
        dataset_name="local-finetune-dataset",
        manifests=[outputs.frame_manifest, outputs.detections_with_images, outputs.low_confidence, outputs.crop_manifest],
        output=outputs.data_card,
        version_paths=[outputs.frame_manifest, outputs.detections_with_images, outputs.low_confidence, outputs.crop_manifest, outputs.annotation_config],
    )
    if train_object_model:
        if base_model_path is None:
            raise ValueError("base_model_path is required when train_object_model is true")
        run_object_training(
            ObjectTrainingConfig(
                data_config=outputs.annotation_config,
                base_model=base_model_path,
                output_dir=output_dir / "object_training",
                epochs=object_epochs,
                image_size=object_image_size,
            )
        )
    return outputs
