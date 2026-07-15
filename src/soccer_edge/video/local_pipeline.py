from dataclasses import dataclass
from pathlib import Path

from soccer_edge.video.yolo_pipeline import run_yolo_detection


@dataclass(frozen=True)
class LocalVideoPipelineResult:
    input_path: Path
    output_dir: Path
    frame_count: int
    table_paths: dict[str, Path]


def run_local_video_pipeline(
    input_path: Path,
    output_dir: Path,
    model_path: str | Path,
    stride: int = 1,
    max_samples: int | None = None,
    confidence_threshold: float = 0.25,
    enforce_rights: bool = True,
    rights_manifest: Path | None = None,
    rights_video_id: str | None = None,
    licensed_root: Path = Path("data/raw/video_licensed"),
) -> LocalVideoPipelineResult:
    """Run the rights-gated YOLO detection pipeline over local footage.

    This is a thin convenience wrapper around ``run_yolo_detection`` that returns the
    produced state-table paths plus the frame count (read back from the detections
    table). The rights gate is enforced exactly as in ``run_yolo_detection``.
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    table_paths = run_yolo_detection(
        input_path=input_path,
        output_dir=output_dir,
        model_path=model_path,
        stride=stride,
        max_samples=max_samples,
        confidence_threshold=confidence_threshold,
        enforce_rights=enforce_rights,
        rights_manifest=rights_manifest,
        rights_video_id=rights_video_id,
        licensed_root=licensed_root,
    )
    detections_path = table_paths.get("detections")
    frame_count = 0
    if detections_path is not None and Path(detections_path).exists():
        import pandas as pd

        frame_count = int(pd.read_parquet(detections_path)["frame_idx"].nunique()) if detections_path.suffix == ".parquet" else int(pd.read_csv(detections_path)["frame_idx"].nunique())
    return LocalVideoPipelineResult(
        input_path=input_path,
        output_dir=output_dir,
        frame_count=frame_count,
        table_paths=table_paths,
    )
