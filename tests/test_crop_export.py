import pandas as pd
import pytest

from soccer_edge.crop_export import clamp_box, crop_filename, export_image_crops
from soccer_edge.media_reader import MissingMediaReaderError


def test_clamp_box() -> None:
    assert clamp_box(-5, -1, 20, 10, width=12, height=8) == (0, 0, 12, 8)


def test_crop_filename() -> None:
    row = pd.Series({"frame_idx": 3, "class_name": "ball carrier"})
    assert crop_filename(row, 2) == "3_2_ball_carrier.jpg"


def test_export_image_crops_requires_image_path(tmp_path) -> None:
    with pytest.raises(ValueError):
        export_image_crops(pd.DataFrame([{"x1": 0, "y1": 0, "x2": 1, "y2": 1}]), tmp_path)


def test_export_image_crops_optional_reader(tmp_path) -> None:
    try:
        import numpy as np
        reader = __import__("cv2")
    except Exception:
        pytest.skip("optional reader dependency is not installed")
    image_path = tmp_path / "frame.jpg"
    reader.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
    frame = pd.DataFrame([{"image_path": str(image_path), "frame_idx": 1, "class_name": "player", "x1": 0, "y1": 0, "x2": 10, "y2": 10}])
    try:
        manifest = export_image_crops(frame, tmp_path / "crops")
    except MissingMediaReaderError:
        pytest.skip("optional reader dependency is not installed")
    assert len(manifest) == 1
    assert (tmp_path / "crops" / "1_0_player.jpg").exists()
