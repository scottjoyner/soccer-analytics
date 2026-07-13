import pandas as pd
import pytest

from soccer_edge.correction_merge import merge_reviewed_corrections, merge_reviewed_corrections_from_tables


def test_merge_reviewed_corrections_updates_and_drops() -> None:
    base = pd.DataFrame(
        [
            {"crop_path": "a.jpg", "class_name": "player", "confidence": 0.2},
            {"crop_path": "b.jpg", "class_name": "ball", "confidence": 0.3},
        ]
    )
    corrections = pd.DataFrame(
        [
            {"crop_path": "a.jpg", "review_action": "correct", "corrected_class_name": "referee"},
            {"crop_path": "b.jpg", "review_action": "drop"},
        ]
    )
    merged = merge_reviewed_corrections(base, corrections, key_columns=["crop_path"])
    assert len(merged) == 1
    assert merged.iloc[0]["class_name"] == "referee"
    assert merged.iloc[0]["review_status"] == "correct"


def test_merge_reviewed_corrections_from_tables(tmp_path) -> None:
    base = tmp_path / "base.csv"
    corrections = tmp_path / "corrections.csv"
    output = tmp_path / "corrected.csv"
    pd.DataFrame([{"crop_path": "a.jpg", "class_name": "player"}]).to_csv(base, index=False)
    pd.DataFrame([{"crop_path": "a.jpg", "review_action": "correct", "corrected_class_name": "ball"}]).to_csv(corrections, index=False)
    path = merge_reviewed_corrections_from_tables(base, corrections, output, key_columns=["crop_path"])
    assert path.exists()
    assert pd.read_csv(path).iloc[0]["class_name"] == "ball"


def test_merge_reviewed_corrections_validates_keys() -> None:
    with pytest.raises(ValueError):
        merge_reviewed_corrections(pd.DataFrame([{"x": 1}]), pd.DataFrame([{"crop_path": "a.jpg"}]), key_columns=["crop_path"])
