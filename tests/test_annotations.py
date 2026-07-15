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
    # Groups are keyed by image when images are present: frame_1/frame_2 train, frame_3 val
    # (sorted, 60% of 3 -> 2 train). A frame's boxes stay together under one image name.
    assert (tmp_path / "yolo" / "images" / "train" / "frame_1.jpg").is_symlink()
    assert (tmp_path / "yolo" / "images" / "train" / "frame_2.jpg").exists()
    assert (tmp_path / "yolo" / "images" / "val" / "frame_3.jpg").exists()
    assert len((tmp_path / "yolo" / "labels" / "train" / "frame_1.txt").read_text(encoding="utf-8").strip().splitlines()) == 2
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


def test_arrange_yolo_dataset_no_image_leakage(tmp_path) -> None:
    shared = tmp_path / "shared.jpg"
    other = tmp_path / "other.jpg"
    shared.write_bytes(b"img-a")
    other.write_bytes(b"img-b")
    # Two frame groups reference the SAME physical image; it must land in exactly one split.
    detections = pd.DataFrame(
        [
            {"frame_idx": 1, "class_name": "player", "x1": 0, "y1": 0, "x2": 20, "y2": 10, "image_path": str(shared)},
            {"frame_idx": 2, "class_name": "ball", "x1": 5, "y1": 5, "x2": 15, "y2": 15, "image_path": str(shared)},
            {"frame_idx": 3, "class_name": "player", "x1": 2, "y1": 2, "x2": 12, "y2": 8, "image_path": str(other)},
        ]
    )
    arrange_yolo_dataset(detections, tmp_path / "yolo", ["player", "ball"], 100, 50, train_fraction=0.5)
    train_links = list((tmp_path / "yolo" / "images" / "train").glob("*.jpg"))
    val_links = list((tmp_path / "yolo" / "images" / "val").glob("*.jpg"))
    all_links = [p.name for p in train_links + val_links]
    assert all_links.count("shared.jpg") == 1, all_links
    assert len(train_links + val_links) == 2


def test_arrange_yolo_dataset_remaps_class_aliases(tmp_path) -> None:
    detections = pd.DataFrame(
        [
            {"frame_idx": 1, "class_name": "person", "x1": 0, "y1": 0, "x2": 20, "y2": 10},
            {"frame_idx": 1, "class_name": "sports ball", "x1": 5, "y1": 5, "x2": 15, "y2": 15},
            {"frame_idx": 2, "class_name": "bird", "x1": 1, "y1": 1, "x2": 9, "y2": 9},
        ]
    )
    aliases = {"person": "player", "sports ball": "ball"}
    paths = arrange_yolo_dataset(detections, tmp_path / "yolo", ["player", "ball"], 100, 50, class_aliases=aliases)
    train_txt = (tmp_path / "yolo" / "labels" / "train" / "1.txt").read_text(encoding="utf-8").strip().splitlines()
    # person->player (id 0) and sports ball->ball (id 1) survive; bird is dropped.
    assert len(train_txt) == 2
    assert train_txt[0].startswith("0 ")
    assert train_txt[1].startswith("1 ")


def test_arrange_yolo_dataset_rejects_bad_train_fraction(tmp_path) -> None:
    detections = pd.DataFrame(
        [
            {"frame_idx": 1, "class_name": "player", "x1": 0, "y1": 0, "x2": 20, "y2": 10},
            {"frame_idx": 2, "class_name": "ball", "x1": 5, "y1": 5, "x2": 15, "y2": 15},
        ]
    )
    import pytest

    with pytest.raises(ValueError, match="train_fraction"):
        arrange_yolo_dataset(detections, tmp_path / "yolo", ["player", "ball"], 100, 50, train_fraction=0.0)
    with pytest.raises(ValueError, match="train_fraction"):
        arrange_yolo_dataset(detections, tmp_path / "yolo", ["player", "ball"], 100, 50, train_fraction=1.5)
