import pytest

from soccer_edge.card_validation import assert_valid_cards, validate_data_card, validate_model_card
from soccer_edge.cards import write_data_card, write_model_card
from soccer_edge.models.bundle import save_bundle


def test_validate_model_and_data_cards(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    save_bundle({"kind": "demo"}, bundle, "demo", "v1", ["x"], {"accuracy": 1.0})
    model_card = write_model_card(bundle, tmp_path / "MODEL_CARD.md")
    data_card = write_data_card("demo", [tmp_path], tmp_path / "DATA_CARD.md")
    assert validate_model_card(model_card).is_valid
    assert validate_data_card(data_card).is_valid
    assert_valid_cards(model_card, data_card)


def test_assert_valid_cards_reports_missing_sections(tmp_path) -> None:
    bad = tmp_path / "MODEL_CARD.md"
    bad.write_text("# Model Card: incomplete", encoding="utf-8")
    with pytest.raises(ValueError):
        assert_valid_cards(model_card=bad)
