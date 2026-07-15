"""Player/ball object-model fine-tuning convenience wrapper.

Thin orchestration over the existing local fine-tune pipeline, pinned to soccer classes
(player, ball). This prepares training data (frames -> detections -> low-confidence crops
-> annotation split -> YOLO data.yaml) and optionally launches ultralytics training.

Only point this at local licensed footage (owned/licensed/compatible_license). Public
video URLs are discovery metadata and must never be used as inputs.
"""

from pathlib import Path

from soccer_edge.local_finetune_pipeline import LocalFinetuneOutputs, run_local_finetune_pipeline

SOCCER_CLASSES = ["player", "ball"]
# YOLO/COCO models emit these names; map them onto the soccer classes so
# real detections are not silently dropped by the class filter.
COCO_TO_SOCCER = {"person": "player", "sports ball": "ball"}


def run_player_ball_finetune(
    input_path: Path,
    base_model_path: Path,
    output_dir: Path,
    object_model_path: Path | None = None,
    calibration_path: Path | None = None,
    stride: int = 5,
    max_frames: int | None = 100,
    train_fraction: float = 0.8,
    threshold: float = 0.5,
    image_width: float = 1920.0,
    image_height: float = 1080.0,
    train_object_model: bool = True,
    object_epochs: int = 50,
    object_image_size: int = 640,
    class_aliases: dict[str, str] | None = None,
) -> LocalFinetuneOutputs:
    return run_local_finetune_pipeline(
        input_path=input_path,
        object_model_path=object_model_path if object_model_path is not None else base_model_path,
        output_dir=output_dir,
        classes=list(SOCCER_CLASSES),
        base_model_path=base_model_path,
        calibration_path=calibration_path,
        stride=stride,
        max_frames=max_frames,
        train_fraction=train_fraction,
        threshold=threshold,
        image_width=image_width,
        image_height=image_height,
        train_object_model=train_object_model,
        object_epochs=object_epochs,
        object_image_size=object_image_size,
        class_aliases=class_aliases if class_aliases is not None else dict(COCO_TO_SOCCER),
    )
