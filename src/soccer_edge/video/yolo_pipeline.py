from dataclasses import asdict
from pathlib import Path

from soccer_edge.media_samples import iter_media_samples
from soccer_edge.media_pitch import add_pitch_point
from soccer_edge.video.detector import YOLODetector
from soccer_edge.video.homography import HomographyTransform
from soccer_edge.video.state_tables import write_video_state_tables


def run_yolo_detection(
    input_path: Path,
    output_dir: Path,
    model_path: str | Path,
    stride: int = 1,
    max_samples: int | None = None,
    confidence_threshold: float = 0.25,
    transform: HomographyTransform | None = None,
) -> dict[str, Path]:
    """Run a YOLO detector over approved local footage and write detection tables.

    Only point this at local licensed footage (owned/licensed/compatible_license).
    Public URLs are discovery metadata only and must never be used as inputs here.
    """

    detector = YOLODetector(model_path, confidence_threshold=confidence_threshold)
    rows: list[object] = []
    for sample in iter_media_samples(input_path, stride=stride, max_samples=max_samples):
        detections = detector.detect(sample.index, sample.data)
        if transform is not None:
            rows.extend(add_pitch_point(asdict(detection), transform) for detection in detections)
        else:
            rows.extend(detections)
    return write_video_state_tables(output_dir=output_dir, detections=rows)
