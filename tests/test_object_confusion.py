import pandas as pd
import pytest

from soccer_edge.object_confusion import confusion_matrix_svg, confusion_matrix_table, write_confusion_outputs


def test_confusion_matrix_table() -> None:
    frame = pd.DataFrame(
        [
            {"actual_class": "player", "predicted_class": "player"},
            {"actual_class": "player", "predicted_class": "ball"},
            {"actual_class": "ball", "predicted_class": "ball"},
        ]
    )
    matrix = confusion_matrix_table(frame)
    assert "actual_class" in matrix.columns
    assert "player" in matrix.columns
    assert "ball" in matrix.columns


def test_confusion_matrix_svg() -> None:
    matrix = pd.DataFrame([{"actual_class": "player", "player": 2, "ball": 1}])
    svg = confusion_matrix_svg(matrix)
    assert "<svg" in svg
    assert "Object Confusion Matrix" in svg


def test_write_confusion_outputs(tmp_path) -> None:
    source = tmp_path / "eval.csv"
    table = tmp_path / "matrix.csv"
    svg = tmp_path / "matrix.svg"
    pd.DataFrame([{"actual_class": "player", "predicted_class": "player"}]).to_csv(source, index=False)
    paths = write_confusion_outputs(source, table, svg)
    assert paths["table"].exists()
    assert paths["svg"].exists()


def test_confusion_matrix_table_validates_columns() -> None:
    with pytest.raises(ValueError):
        confusion_matrix_table(pd.DataFrame([{"x": 1}]))
