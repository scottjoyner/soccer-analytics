import pandas as pd

from soccer_edge.contact_sheet import contact_sheet_html, write_contact_sheet


def test_contact_sheet_html() -> None:
    frame = pd.DataFrame([{"crop_path": "crop.jpg", "class_name": "player", "confidence": 0.7, "frame_idx": 1}])
    html = contact_sheet_html(frame)
    assert "Crop Review" in html
    assert "crop.jpg" in html
    assert "player" in html


def test_write_contact_sheet(tmp_path) -> None:
    source = tmp_path / "crops.csv"
    output = tmp_path / "review.html"
    pd.DataFrame([{"crop_path": "crop.jpg", "class_name": "ball"}]).to_csv(source, index=False)
    path = write_contact_sheet(source, output)
    assert path.exists()
    assert "ball" in path.read_text(encoding="utf-8")
