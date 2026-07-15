from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlanValidationResult:
    ok: bool
    missing_paths: list[str]


def shell_quote(path: Path | str) -> str:
    text = str(path)
    if not text:
        return "''"
    if all(character.isalnum() or character in "._/-" for character in text):
        return text
    return "'" + text.replace("'", "'\\''") + "'"


def validate_local_finetune_inputs(
    input_path: Path,
    object_model_path: Path,
    calibration_path: Path | None = None,
) -> PlanValidationResult:
    required = [input_path, object_model_path]
    if calibration_path is not None:
        required.append(calibration_path)
    missing = [str(path) for path in required if not path.exists()]
    return PlanValidationResult(ok=not missing, missing_paths=missing)


def local_finetune_shell_plan(
    input_path: Path,
    object_model_path: Path,
    output_dir: Path,
    classes: str = "player,ball",
    calibration_path: Path | None = None,
    stride: int = 5,
    max_frames: int | None = 100,
    threshold: float = 0.5,
    train_fraction: float = 0.8,
) -> str:
    frame_manifest = output_dir / "frame_manifest.csv"
    detections = output_dir / "video_model" / "detections.parquet"
    detections_with_images = output_dir / "detections_with_images.csv"
    low_confidence = output_dir / "low_confidence.csv"
    crop_manifest = output_dir / "crop_manifest.csv"
    annotations = output_dir / "annotations"
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"soccer-edge video export-frames --input {shell_quote(input_path)} --output-dir {shell_quote(output_dir / 'frames')} --manifest-output {shell_quote(frame_manifest)} --stride {stride}"
        + (f" --max-frames {max_frames}" if max_frames is not None else ""),
        f"soccer-edge video process-local-model --input {shell_quote(input_path)} --model-path {shell_quote(object_model_path)} --output-dir {shell_quote(output_dir / 'video_model')} --stride {stride}"
        + (f" --max-samples {max_frames}" if max_frames is not None else "")
        + (f" --calibration {shell_quote(calibration_path)}" if calibration_path is not None else ""),
        f"soccer-edge video attach-frame-images --detections {shell_quote(detections)} --frame-manifest {shell_quote(frame_manifest)} --output {shell_quote(detections_with_images)}",
        f"soccer-edge video sample-low-confidence --source {shell_quote(detections_with_images)} --output {shell_quote(low_confidence)} --threshold {threshold}",
        f"soccer-edge video export-crops --source {shell_quote(low_confidence)} --output-dir {shell_quote(output_dir / 'crops')} --manifest-output {shell_quote(crop_manifest)} --image-path-column image_path",
        f"soccer-edge video contact-sheet --source {shell_quote(crop_manifest)} --output {shell_quote(output_dir / 'crop_review.html')}",
        f"soccer-edge video export-annotations --source {shell_quote(detections_with_images)} --output-dir {shell_quote(annotations)} --classes {shell_quote(classes)}",
        f"soccer-edge video split-annotations --source {shell_quote(detections_with_images)} --train-output {shell_quote(annotations / 'train.csv')} --val-output {shell_quote(annotations / 'val.csv')} --train-fraction {train_fraction}",
        f"soccer-edge video prepare-object-dataset --source {shell_quote(detections_with_images)} --output-dir {shell_quote(annotations / 'yolo')} --classes {shell_quote(classes)} --train-fraction {train_fraction}",
        f"soccer-edge model auto-data-card --dataset-name local-finetune-dataset --manifests {shell_quote(str(frame_manifest) + ',' + str(detections_with_images) + ',' + str(crop_manifest))} --output {shell_quote(output_dir / 'DATA_CARD.md')}",
    ]
    return "\n".join(commands) + "\n"


def write_local_finetune_shell_plan(
    output_path: Path,
    input_path: Path,
    object_model_path: Path,
    output_dir: Path,
    classes: str = "player,ball",
    calibration_path: Path | None = None,
    stride: int = 5,
    max_frames: int | None = 100,
    threshold: float = 0.5,
    train_fraction: float = 0.8,
    validate_inputs: bool = False,
) -> Path:
    if validate_inputs:
        validation = validate_local_finetune_inputs(input_path, object_model_path, calibration_path)
        if not validation.ok:
            raise FileNotFoundError(f"missing required paths: {validation.missing_paths}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        local_finetune_shell_plan(
            input_path=input_path,
            object_model_path=object_model_path,
            output_dir=output_dir,
            classes=classes,
            calibration_path=calibration_path,
            stride=stride,
            max_frames=max_frames,
            threshold=threshold,
            train_fraction=train_fraction,
        ),
        encoding="utf-8",
    )
    output_path.chmod(0o755)
    return output_path
