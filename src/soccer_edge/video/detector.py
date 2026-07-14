from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    frame_idx: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class Detector:
    def detect_frame(self, frame_idx: int, frame: object) -> list[Detection]:
        raise NotImplementedError

    def detect(self, frame_idx: int, frame: object) -> list[Detection]:
        """Convenience alias used by some pipelines; defaults to detect_frame."""

        return self.detect_frame(frame_idx=frame_idx, frame=frame)


class NullDetector(Detector):
    def detect_frame(self, frame_idx: int, frame: object) -> list[Detection]:
        return []


def detections_from_rows(rows: Iterable[object], frame_idx: int, confidence_threshold: float = 0.25) -> list[Detection]:
    """Map raw model rows (class_name/confidence/x1..y2) into Detection objects."""

    detections: list[Detection] = []
    for row in rows:
        confidence = float(row.get("confidence", 0.0))
        if confidence < confidence_threshold:
            continue
        detections.append(
            Detection(
                frame_idx=frame_idx,
                class_name=str(row.get("class_name", "object")),
                confidence=confidence,
                x1=float(row.get("x1", 0.0)),
                y1=float(row.get("y1", 0.0)),
                x2=float(row.get("x2", 0.0)),
                y2=float(row.get("y2", 0.0)),
            )
        )
    return detections


class YOLODetector(Detector):
    """Detector backed by an ultralytics YOLO model (e.g. yolov8n.pt or a fine-tuned .pt).

    The ultralytics dependency is imported lazily so this module stays importable in
    environments without the optional ML stack.
    """

    def __init__(self, model_path: str | Path, confidence_threshold: float = 0.25) -> None:
        from soccer_edge.object_model import LocalObjectRunner

        self._runner = LocalObjectRunner(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_frame(self, frame_idx: int, frame: object) -> list[Detection]:
        rows = self._runner(frame)
        return detections_from_rows(rows, frame_idx, confidence_threshold=self.confidence_threshold)

