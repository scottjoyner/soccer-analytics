from dataclasses import asdict
from pathlib import Path

from soccer_edge.media_samples import iter_media_samples
from soccer_edge.media_pitch import add_pitch_point
from soccer_edge.video.batch_runner import assert_processable
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
    enforce_rights: bool = True,
    rights_manifest: Path | None = None,
    rights_video_id: str | None = None,
    licensed_root: Path = Path("data/raw/video_licensed"),
) -> dict[str, Path]:
    """Run a YOLO detector over approved local footage and write detection tables.

    By default this enforces the rights gate: the footage must correspond to an
    approved, rights-referenced manifest row (owned/licensed/compatible_license)
    under ``licensed_root``. Public URLs are discovery metadata only and must never
    be used as inputs. Callers processing synthetic or already-approved frames may
    pass ``enforce_rights=False`` with an explicit comment justifying the bypass.
    """

    if enforce_rights:
        if rights_manifest is None or rights_video_id is None:
            raise ValueError(
                "run_yolo_detection requires an approved manifest row (rights_manifest + "
                "rights_video_id) unless enforce_rights=False is explicitly passed for "
                "synthetic/pre-approved frames."
            )
        assert_processable(rights_manifest, rights_video_id, Path(input_path), Path(licensed_root))

    detector = YOLODetector(model_path, confidence_threshold=confidence_threshold)
    rows: list[object] = []
    for sample in iter_media_samples(input_path, stride=stride, max_samples=max_samples):
        detections = detector.detect(sample.index, sample.data)
        if transform is not None:
            rows.extend(add_pitch_point(asdict(detection), transform) for detection in detections)
        else:
            rows.extend(detections)
    return write_video_state_tables(output_dir=output_dir, detections=rows)
