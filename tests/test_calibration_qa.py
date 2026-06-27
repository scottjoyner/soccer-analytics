import json

import pandas as pd

from soccer_edge.calibration_qa import projection_qa_frame, projection_qa_svg, write_projection_qa_csv, write_projection_qa_svg
from soccer_edge.video.homography import HomographyTransform
import numpy as np


def test_projection_qa_frame() -> None:
    frame = projection_qa_frame([(0.0, 0.0), (1.0, 1.0)], [(0.0, 0.0), (1.0, 1.0)], HomographyTransform(np.eye(3)))
    assert list(frame["error_m"]) == [0.0, 0.0]


def test_projection_qa_svg() -> None:
    frame = pd.DataFrame([{"expected_x_m": 0.0, "expected_y_m": 0.0, "projected_x_m": 1.0, "projected_y_m": 1.0}])
    svg = projection_qa_svg(frame)
    assert "<svg" in svg
    assert "circle" in svg


def test_write_projection_qa_files(tmp_path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "pixel_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "pitch_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
            }
        ),
        encoding="utf-8",
    )
    csv_path = write_projection_qa_csv(calibration, tmp_path / "qa.csv")
    svg_path = write_projection_qa_svg(calibration, tmp_path / "qa.svg")
    assert csv_path.exists()
    assert svg_path.exists()
