from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationPoint:
    pixel_x: float
    pixel_y: float
    pitch_x_m: float
    pitch_y_m: float


@dataclass(frozen=True)
class PitchCalibration:
    video_id: str
    pitch_length_m: float
    pitch_width_m: float
    points: tuple[CalibrationPoint, ...]
    confidence: float = 0.0
    notes: str = ""

    def validate(self) -> None:
        if len(self.points) < 4:
            raise ValueError("At least four calibration points are required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
