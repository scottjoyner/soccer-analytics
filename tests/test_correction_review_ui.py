import pandas as pd
import pytest

from soccer_edge.correction_review_ui import correction_review_html, correction_template, write_correction_review_assets


def test_correction_template() -> None:
    frame = pd.DataFrame([{"crop_path": "a.jpg", "class_name": "player"}])
    template = correction_template(frame, ["crop_path"])
    assert "review_action" in template.columns
    assert "corrected_class_name" in template.columns


def test_correction_review_html() -> None:
    frame = pd.DataFrame([{"crop_path": "a.jpg", "class_name": "player", "x1": 1, "y1": 2, "x2": 3, "y2": 4}])
    html = correction_review_html(frame, ["crop_path"])
    assert "Correction Review" in html
    assert "corrected_x1" in html
    assert "<img" in html


def test_write_correction_review_assets(tmp_path) -> None:
    source = tmp_path / "rows.csv"
    html = tmp_path / "review.html"
    template = tmp_path / "corrections.csv"
    pd.DataFrame([{"crop_path": "a.jpg", "class_name": "player"}]).to_csv(source, index=False)
    paths = write_correction_review_assets(source, html, template, ["crop_path"])
    assert paths["html"].exists()
    assert paths["template"].exists()


def test_correction_template_validates_key() -> None:
    with pytest.raises(ValueError):
        correction_template(pd.DataFrame([{"x": 1}]), ["crop_path"])
