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


def run_local_video_pipeline(input_path: Path, output_dir: Path, frame_count: int = 0) -> LocalVideoPipelineResult:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
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
