import pandas as pd

from soccer_edge.auto_data_card import auto_data_card_markdown, manifest_stats, write_auto_data_card


def test_manifest_stats(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"rights_status": "owned", "class_name": "player"}]).to_csv(manifest, index=False)
    stats = manifest_stats(manifest)
    assert stats["rows"] == 1
    assert stats["rights_statuses"] == "owned"
    assert stats["classes"] == "player"


def test_auto_data_card_markdown(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"rights_status": "owned"}]).to_csv(manifest, index=False)
    text = auto_data_card_markdown("demo", [manifest])
    assert "Data Card" in text
    assert "Source catalog" in text
    assert "Asset versions" in text


def test_write_auto_data_card(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "DATA_CARD.md"
    pd.DataFrame([{"rights_status": "owned"}]).to_csv(manifest, index=False)
    path = write_auto_data_card("demo", [manifest], output)
    assert path.exists()
