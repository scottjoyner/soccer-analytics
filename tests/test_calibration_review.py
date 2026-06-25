import pandas as pd
import pytest

from soccer_edge.evaluation.calibration_review import build_calibration_review, probability_columns, write_calibration_review


def test_probability_columns_sorted() -> None:
    frame = pd.DataFrame(columns=["prob_2", "prob_0", "prob_1"])
    assert probability_columns(frame) == ["prob_0", "prob_1", "prob_2"]


def test_build_calibration_review() -> None:
    frame = pd.DataFrame(
        [
            {"label": 0, "prob_0": 0.8, "prob_1": 0.1, "prob_2": 0.1},
            {"label": 1, "prob_0": 0.1, "prob_1": 0.8, "prob_2": 0.1},
        ]
    )
    review = build_calibration_review(frame, num_bins=2)
    assert review.row_count == 2
    assert review.metrics.accuracy == 1.0


def test_write_calibration_review(tmp_path) -> None:
    frame = pd.DataFrame([{"label": 0, "prob_0": 0.8, "prob_1": 0.1, "prob_2": 0.1}])
    paths = write_calibration_review(frame, tmp_path)
    assert paths["metrics"].exists()
    assert paths["calibration"].exists()
    assert paths["rows"].exists()


def test_build_calibration_review_requires_label() -> None:
    with pytest.raises(ValueError):
        build_calibration_review(pd.DataFrame([{"prob_0": 1.0}]))
