import pandas as pd

from soccer_edge.annotations import (
    arrange_yolo_dataset,
    normalized_box_line,
    write_detection_annotations,
)


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


def test_arrange_yolo_dataset(tmp_path) -> None:
    frame1 = tmp_path / "frame_1.jpg"
    frame2 = tmp_path / "frame_2.jpg"
    frame3 = tmp_path / "frame_3.jpg"
    for image in (frame1, frame2, frame3):
        image.write_bytes(b"fake-image")
    detections = pd.DataFrame(
        [
            {"frame_idx": 1, "class_name": "player", "x1": 0, "y1": 0, "x2": 20, "y2": 10, "image_path": str(frame1)},
            {"frame_idx": 1, "class_name": "ball", "x1": 10, "y1": 10, "x2": 20, "y2": 20, "image_path": str(frame1)},
            {"frame_idx": 2, "class_name": "player", "x1": 5, "y1": 5, "x2": 25, "y2": 15, "image_path": str(frame2)},
            {"frame_idx": 3, "class_name": "player", "x1": 2, "y1": 2, "x2": 12, "y2": 8, "image_path": str(frame3)},
        ]
    )
    paths = arrange_yolo_dataset(detections, tmp_path / "yolo", ["player", "ball"], 100, 50, train_fraction=0.6)
    for name in ("images_train", "images_val", "labels_train", "labels_val", "classes", "config"):
        assert paths[name].exists(), name
    # Frame groups are not split: frames 1-2 train, frame 3 val (sorted, 60% of 3 -> 2 train).
    assert (tmp_path / "yolo" / "images" / "train" / "1.jpg").is_symlink()
    assert (tmp_path / "yolo" / "images" / "train" / "2.jpg").exists()
    assert (tmp_path / "yolo" / "images" / "val" / "3.jpg").exists()
    assert len((tmp_path / "yolo" / "labels" / "train" / "1.txt").read_text(encoding="utf-8").strip().splitlines()) == 2
    yaml_text = paths["config"].read_text(encoding="utf-8")
    assert "train: images/train" in yaml_text
    assert "val: images/val" in yaml_text


def test_arrange_yolo_dataset_missing_images_skips_links(tmp_path) -> None:
    detections = pd.DataFrame(
        [
            {"frame_idx": 1, "class_name": "player", "x1": 0, "y1": 0, "x2": 20, "y2": 10},
        ]
    )
    paths = arrange_yolo_dataset(detections, tmp_path / "yolo", ["player", "ball"], 100, 50)
    assert paths["labels_train"].joinpath("1.txt").exists()
    assert not list((tmp_path / "yolo" / "images" / "train").glob("*.jpg"))
    assert paths["classes"].exists()
