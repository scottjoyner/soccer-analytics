import json
import numpy as np
import pytest

from soccer_edge.video.calibration_io import load_homography, point_pairs, read_mapping


def test_load_homography_json(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "pixel_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "pitch_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
            }
        ),
        encoding="utf-8",
    )
    transform = load_homography(path)
    point = transform.transform_pixel(0.5, 0.5)
    assert point is not None
    assert np.isclose(point.x_m, 0.5)
    assert np.isclose(point.y_m, 0.5)


def test_point_pairs_points_shape() -> None:
    pixels, pitch = point_pairs(
        {"points": [{"pixel": [0, 0], "pitch": [0, 0]}, {"pixel": [1, 0], "pitch": [1, 0]}, {"pixel": [1, 1], "pitch": [1, 1]}, {"pixel": [0, 1], "pitch": [0, 1]}]}
    )
    assert pixels[0] == (0.0, 0.0)
    assert pitch[-1] == (0.0, 1.0)


def test_read_mapping_rejects_unknown_suffix(tmp_path) -> None:
    path = tmp_path / "calibration.txt"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        read_mapping(path)
