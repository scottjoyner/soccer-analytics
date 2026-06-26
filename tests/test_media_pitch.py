import numpy as np

from soccer_edge.media_pitch import add_pitch_point, add_pitch_points
from soccer_edge.media_inference import MediaBox
from soccer_edge.video.homography import HomographyTransform


def test_add_pitch_point() -> None:
    transform = HomographyTransform(matrix=np.eye(3))
    row = MediaBox(frame_idx=1, timestamp_seconds=0.5, class_name="player", confidence=0.9, x1=0.0, y1=0.0, x2=2.0, y2=4.0)
    output = add_pitch_point(row, transform)
    assert output["pixel_center_x"] == 1.0
    assert output["pixel_center_y"] == 2.0
    assert output["pitch_x_m"] == 1.0
    assert output["pitch_y_m"] == 2.0


def test_add_pitch_points_without_transform_returns_rows() -> None:
    rows = [{"x1": 0.0, "y1": 0.0, "x2": 2.0, "y2": 2.0}]
    assert add_pitch_points(rows, None) == rows
