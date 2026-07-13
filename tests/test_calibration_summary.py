import pandas as pd
import pytest

from soccer_edge.calibration_summary import calibration_error_stats, calibration_summary_markdown, write_calibration_summary


def test_calibration_error_stats() -> None:
    stats = calibration_error_stats(pd.DataFrame([{"error_m": 0.0}, {"error_m": 2.0}]))
    assert stats["mean_error_m"] == 1.0
    assert stats["max_error_m"] == 2.0


def test_calibration_summary_markdown() -> None:
    text = calibration_summary_markdown(pd.DataFrame([{"error_m": 0.5}]))
    assert "Calibration QA Summary" in text
    assert "Mean error" in text


def test_write_calibration_summary(tmp_path) -> None:
    source = tmp_path / "qa.csv"
    output = tmp_path / "qa.md"
    pd.DataFrame([{"error_m": 0.5}]).to_csv(source, index=False)
    path = write_calibration_summary(source, output)
    assert path.exists()


def test_calibration_error_stats_requires_error_column() -> None:
    with pytest.raises(ValueError):
        calibration_error_stats(pd.DataFrame([{"x": 1}]))
