import pandas as pd
import pytest

from soccer_edge.frame_join import attach_image_paths, attach_image_paths_from_tables


def test_attach_image_paths() -> None:
    detections = pd.DataFrame([{"frame_idx": 1, "class_name": "player"}, {"frame_idx": 2, "class_name": "ball"}])
    manifest = pd.DataFrame([{"frame_idx": 1, "image_path": "f1.jpg"}, {"frame_idx": 2, "image_path": "f2.jpg"}])
    joined = attach_image_paths(detections, manifest)
    assert list(joined["image_path"]) == ["f1.jpg", "f2.jpg"]


def test_attach_image_paths_from_tables(tmp_path) -> None:
    detections = tmp_path / "detections.csv"
    manifest = tmp_path / "frames.csv"
    output = tmp_path / "joined.csv"
    pd.DataFrame([{"frame_idx": 1, "class_name": "player"}]).to_csv(detections, index=False)
    pd.DataFrame([{"frame_idx": 1, "image_path": "f1.jpg"}]).to_csv(manifest, index=False)
    path = attach_image_paths_from_tables(detections, manifest, output)
    assert path.exists()
    assert pd.read_csv(path).iloc[0]["image_path"] == "f1.jpg"


def test_attach_image_paths_validates_columns() -> None:
    with pytest.raises(ValueError):
        attach_image_paths(pd.DataFrame([{"x": 1}]), pd.DataFrame([{"frame_idx": 1, "image_path": "f1.jpg"}]))
