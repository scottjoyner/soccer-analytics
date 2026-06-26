import pandas as pd
import pytest

from soccer_edge.active_sampling import select_low_confidence_rows, write_low_confidence_rows


def test_select_low_confidence_rows() -> None:
    frame = pd.DataFrame([{"id": 1, "confidence": 0.7}, {"id": 2, "confidence": 0.2}, {"id": 3, "confidence": 0.4}])
    selected = select_low_confidence_rows(frame, threshold=0.5)
    assert list(selected["id"]) == [2, 3]


def test_write_low_confidence_rows(tmp_path) -> None:
    source = tmp_path / "detections.csv"
    output = tmp_path / "review.csv"
    pd.DataFrame([{"id": 1, "confidence": 0.2}, {"id": 2, "confidence": 0.9}]).to_csv(source, index=False)
    path = write_low_confidence_rows(source, output, threshold=0.5)
    assert path.exists()
    assert len(pd.read_csv(path)) == 1


def test_select_low_confidence_requires_column() -> None:
    with pytest.raises(ValueError):
        select_low_confidence_rows(pd.DataFrame([{"x": 1}]))
