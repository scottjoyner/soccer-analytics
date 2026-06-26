from soccer_edge.cards import write_data_card, write_model_card
from soccer_edge.models.bundle import save_bundle


def test_write_model_card(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    save_bundle({"kind": "demo"}, bundle, "demo", "v1", ["x"], {"accuracy": 1.0})
    path = write_model_card(bundle, tmp_path / "MODEL_CARD.md")
    assert path.exists()
    assert "Model Card" in path.read_text(encoding="utf-8")


def test_write_data_card(tmp_path) -> None:
    path = write_data_card("demo", [tmp_path / "source.csv"], tmp_path / "DATA_CARD.md")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Data Card" in text
    assert "Rights status" in text
