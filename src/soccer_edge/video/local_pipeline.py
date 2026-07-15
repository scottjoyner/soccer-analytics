from dataclasses import dataclass
from pathlib import Path

from soccer_edge.video.detector import NullDetector
from soccer_edge.video.state_tables import write_video_state_tables


@dataclass(frozen=True)
class LocalVideoPipelineResult:
    input_path: Path
    output_dir: Path
    frame_count: int
    table_paths: dict[str, Path]


def run_local_video_pipeline(input_path: Path, output_dir: Path, frame_count: int = 1) -> LocalVideoPipelineResult:
    """Synthetic detection generator for tests/demos.

    NOTE: this does NOT read or process the video at ``input_path``. It emits
    ``frame_count`` deterministic placeholder detections via ``NullDetector`` and
    writes them as state tables. Real footage processing goes through
    ``run_yolo_detection`` (which enforces the rights gate). ``input_path`` is only
    checked for existence so callers notice a wrong path; its contents are unused.
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    detector = NullDetector()
    detections = []
    for frame_idx in range(frame_count):
        detections.extend(detector.detect(frame_idx=frame_idx, frame=None))
    table_paths = write_video_state_tables(output_dir=output_dir, detections=detections)
    return LocalVideoPipelineResult(
        input_path=input_path,
        output_dir=output_dir,
        frame_count=frame_count,
        table_paths=table_paths,
    )
