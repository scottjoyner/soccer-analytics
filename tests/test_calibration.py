import pytest

from soccer_edge.video.calibration import CalibrationPoint, PitchCalibration


def test_pitch_calibration_requires_four_points() -> None:
    calibration = PitchCalibration(
        video_id="clip",
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        points=(CalibrationPoint(0, 0, 0, 0),),
        confidence=0.5,
    )
    with pytest.raises(ValueError):
        calibration.validate()


def test_pitch_calibration_accepts_valid_points() -> None:
    calibration = PitchCalibration(
        video_id="clip",
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        points=(
            CalibrationPoint(0, 0, 0, 0),
            CalibrationPoint(100, 0, 105, 0),
            CalibrationPoint(100, 100, 105, 68),
            CalibrationPoint(0, 100, 0, 68),
        ),
        confidence=0.8,
    )
    calibration.validate()
