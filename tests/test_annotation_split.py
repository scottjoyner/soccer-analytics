import pandas as pd
import pytest

from soccer_edge.annotation_split import split_annotation_rows, write_annotation_split


def test_split_annotation_rows_by_frame() -> None:
    frame = pd.DataFrame([{"frame_idx": 1, "x": 1}, {"frame_idx": 2, "x": 2}, {"frame_idx": 3, "x": 3}])
    train, val = split_annotation_rows(frame, train_fraction=0.67)
    assert set(train["frame_idx"]) == {1, 2}
    assert set(val["frame_idx"]) == {3}


def test_write_annotation_split(tmp_path) -> None:
    source = tmp_path / "rows.csv"
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    pd.DataFrame([{"frame_idx": 1}, {"frame_idx": 2}]).to_csv(source, index=False)
    paths = write_annotation_split(source, train, val, train_fraction=0.5)
    assert paths["train"].exists()
    assert paths["val"].exists()


def test_split_annotation_rows_validates_fraction() -> None:
    with pytest.raises(ValueError):
        split_annotation_rows(pd.DataFrame([{"x": 1}]), train_fraction=1.0)
