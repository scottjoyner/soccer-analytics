import pytest

from soccer_edge.annotation_dataset import AnnotationDatasetConfig, annotation_dataset_yaml, write_annotation_dataset_config


def test_annotation_dataset_yaml() -> None:
    text = annotation_dataset_yaml(AnnotationDatasetConfig(root="data", train_images="images/train", val_images="images/val", class_names=["player", "ball"]))
    assert "path: data" in text
    assert "0: player" in text
    assert "1: ball" in text


def test_write_annotation_dataset_config(tmp_path) -> None:
    path = write_annotation_dataset_config(
        AnnotationDatasetConfig(root=tmp_path, train_images=tmp_path / "train", val_images=tmp_path / "val", class_names=["player"]),
        tmp_path / "data.yaml",
    )
    assert path.exists()
    assert "names" in path.read_text(encoding="utf-8")


def test_write_annotation_dataset_config_requires_classes(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_annotation_dataset_config(AnnotationDatasetConfig(root=tmp_path, train_images=tmp_path, val_images=tmp_path, class_names=[]), tmp_path / "bad.yaml")
