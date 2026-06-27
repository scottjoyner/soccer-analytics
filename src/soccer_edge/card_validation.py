from dataclasses import dataclass
from pathlib import Path


REQUIRED_MODEL_CARD_SECTIONS = ["# Model Card", "## Intended use", "## Features", "## Metrics", "## Limitations"]
REQUIRED_DATA_CARD_SECTIONS = ["# Data Card", "Rights status", "## Sources", "## Lineage", "## Restrictions"]


@dataclass(frozen=True)
class CardValidationResult:
    path: Path
    missing_sections: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.missing_sections


def validate_card(path: Path, required_sections: list[str]) -> CardValidationResult:
    text = path.read_text(encoding="utf-8")
    missing = [section for section in required_sections if section not in text]
    return CardValidationResult(path=path, missing_sections=missing)


def validate_model_card(path: Path) -> CardValidationResult:
    return validate_card(path, REQUIRED_MODEL_CARD_SECTIONS)


def validate_data_card(path: Path) -> CardValidationResult:
    return validate_card(path, REQUIRED_DATA_CARD_SECTIONS)


def assert_valid_cards(model_card: Path | None = None, data_card: Path | None = None) -> None:
    results = []
    if model_card is not None:
        results.append(validate_model_card(model_card))
    if data_card is not None:
        results.append(validate_data_card(data_card))
    failures = [result for result in results if not result.is_valid]
    if failures:
        details = "; ".join(f"{failure.path}: missing {failure.missing_sections}" for failure in failures)
        raise ValueError(details)
