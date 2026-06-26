import pandas as pd
import pytest

from soccer_edge.models.splits import SplitConfig, add_time_split, split_frames


def test_add_time_split() -> None:
    frame = pd.DataFrame(
        [
            {"match_id": "a", "match_date": "2025-01-01"},
            {"match_id": "b", "match_date": "2025-06-01"},
            {"match_id": "c", "match_date": "2026-01-01"},
        ]
    )
    output = add_time_split(frame, SplitConfig(train_end="2025-03-01", validation_end="2025-12-31"))
    assert list(output["split"]) == ["train", "validation", "test"]


def test_split_frames() -> None:
    frame = pd.DataFrame(
        [
            {"match_id": "a", "match_date": "2025-01-01"},
            {"match_id": "b", "match_date": "2026-01-01"},
        ]
    )
    splits = split_frames(frame, SplitConfig(train_end="2025-03-01", validation_end="2025-12-31"))
    assert "train" in splits
    assert "test" in splits


def test_add_time_split_requires_ordered_dates() -> None:
    with pytest.raises(ValueError):
        add_time_split(pd.DataFrame([{"match_date": "2025-01-01"}]), SplitConfig("2025-02-01", "2025-01-01"))
