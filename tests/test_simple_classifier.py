import pandas as pd
import pytest

from soccer_edge.models.simple_classifier import fit_simple_classifier


def test_fit_simple_classifier(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"x": 0.0, "y": 0},
            {"x": 1.0, "y": 1},
            {"x": 2.0, "y": 1},
            {"x": -1.0, "y": 0},
        ]
    )
    paths = fit_simple_classifier(frame, ["x"], "y", tmp_path / "model")
    assert paths["model"].exists()
    assert paths["metadata"].exists()


def test_fit_simple_classifier_requires_columns(tmp_path) -> None:
    with pytest.raises(ValueError):
        fit_simple_classifier(pd.DataFrame([{"x": 1.0}]), ["missing"], "y", tmp_path)
