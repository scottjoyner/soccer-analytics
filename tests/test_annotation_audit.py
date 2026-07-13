import pandas as pd
import pytest

from soccer_edge.annotation_audit import annotation_audit_tables, count_by_column, write_annotation_audit


def test_annotation_audit_tables() -> None:
    frame = pd.DataFrame(
        [
            {"frame_idx": 1, "class_name": "player", "split": "train"},
            {"frame_idx": 1, "class_name": "ball", "split": "train"},
            {"frame_idx": 2, "class_name": "player", "split": "val"},
        ]
    )
    tables = annotation_audit_tables(frame)
    assert tables["by_class"].iloc[0]["class_name"] == "player"
    assert "by_split_class" in tables


def test_write_annotation_audit(tmp_path) -> None:
    source = tmp_path / "annotations.csv"
    pd.DataFrame([{"frame_idx": 1, "class_name": "player"}]).to_csv(source, index=False)
    paths = write_annotation_audit(source, tmp_path / "audit")
    assert paths["by_class"].exists()
    assert paths["by_frame"].exists()


def test_count_by_column_validates_column() -> None:
    with pytest.raises(ValueError):
        count_by_column(pd.DataFrame([{"x": 1}]), "class_name")
