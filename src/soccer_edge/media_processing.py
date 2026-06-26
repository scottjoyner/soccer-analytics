from collections.abc import Callable
from pathlib import Path

from soccer_edge.media_pitch import add_pitch_points
from soccer_edge.media_samples import MediaSample, iter_media_samples
from soccer_edge.video.homography import HomographyTransform
from soccer_edge.video.state_tables import write_video_state_tables

MediaCallback = Callable[[MediaSample], list[object]]


def run_media_processing_loop(
    input_path: Path,
    output_dir: Path,
    callback: MediaCallback,
    stride: int = 1,
    max_samples: int | None = None,
    transform: HomographyTransform | None = None,
) -> dict[str, Path]:
    rows: list[object] = []
    for sample in iter_media_samples(input_path, stride=stride, max_samples=max_samples):
        rows.extend(callback(sample))
    rows = add_pitch_points(rows, transform)
    return write_video_state_tables(output_dir=output_dir, detections=rows)
