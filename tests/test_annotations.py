import pandas as pd

from soccer_edge.annotations import normalized_box_line, write_detection_annotations


def test_normalized_box_line() -> None:
    row = pd.Series({"class_name": "player", "x1": 0, "y1": 0, "x2": 20, "y2": 10})
    line = normalized_box_line(row, {"player": 0}, image_width=100, image_height=50)
    assert line == "0 0.100000 0.100000 0.200000 0.200000"


def test_write_detection_annotations(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"frame_idx": 1, "class_name": "player", "x1": 0, "y1": 0, "x2": 20, "y2": 10},
            {"frame_idx": 1, "class_name": "ball", "x1": 10, "y1": 10, "x2": 20, "y2": 20},
        ]
    )
    paths = write_detection_annotations(frame, tmp_path, ["player", "ball"], 100, 50)
    assert paths["1"].exists()
    assert paths["classes"].exists()
    assert len(paths["1"].read_text(encoding="utf-8").strip().splitlines()) == 2
