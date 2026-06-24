from dataclasses import dataclass


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


class NullDetector(Detector):
    def detect_frame(self, frame_idx: int, frame: object) -> list[Detection]:
        return []
