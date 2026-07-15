from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from soccer_edge.card_validation import assert_valid_cards


@dataclass(frozen=True)
class PromotionGateResult:
    ok: bool
    checks: dict[str, bool]
    notes: list[str]


def nonempty_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def nonempty_table(path: Path) -> bool:
    if not nonempty_file(path):
        return False
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return len(frame) > 0


def audit_dir_has_outputs(path: Path) -> bool:
    return path.exists() and any(nonempty_table(file_path) for file_path in path.glob("*.csv"))


def object_metrics_pass(path: Path, min_f1: float = 0.0) -> bool:
    if not nonempty_table(path):
        return False
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "f1" not in frame.columns:
        return False
    return bool((frame["f1"].fillna(0.0) >= min_f1).all())


def read_metrics_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def beats_majority_baseline(
    path: Path | None,
    majority_baseline_rate: float = 0.0,
    min_accuracy_lift: float = 0.0,
) -> tuple[bool, list[str]]:
    if path is None or not nonempty_file(path):
        return True, []
    frame = read_metrics_table(path)
    if "accuracy" not in frame.columns:
        return True, ["no accuracy column; skipping baseline check"]
    accuracy = float(frame["accuracy"].fillna(0.0).mean())
    required = majority_baseline_rate + min_accuracy_lift
    if accuracy < required:
        return (
            False,
            [
                f"accuracy {accuracy:.4f} below required {required:.4f} "
                f"(majority {majority_baseline_rate:.4f} + lift {min_accuracy_lift:.4f})"
            ],
        )
    return True, []


def brier_within_threshold(path: Path | None, max_brier: float | None = None) -> tuple[bool, list[str]]:
    if path is None or max_brier is None or not nonempty_file(path):
        return True, []
    frame = read_metrics_table(path)
    if "brier" not in frame.columns:
        return True, ["no brier column; skipping calibration check"]
    brier = float(frame["brier"].fillna(1.0).mean())
    if brier > max_brier:
        return False, [f"mean brier {brier:.4f} above max {max_brier:.4f}"]
    return True, []


def run_promotion_gate(
    model_card_path: Path | None,
    data_card_path: Path | None,
    dataset_versions_path: Path,
    audit_dir: Path,
    object_metrics_path: Path,
    min_f1: float = 0.0,
    predictive_metrics_path: Path | None = None,
    majority_baseline_rate: float = 0.0,
    min_accuracy_lift: float = 0.0,
    max_brier: float | None = None,
) -> PromotionGateResult:
    notes: list[str] = []
    checks: dict[str, bool] = {}
    try:
        assert_valid_cards(model_card_path, data_card_path)
        checks["cards_valid"] = True
    except Exception as exc:
        checks["cards_valid"] = False
        notes.append(f"card validation failed: {exc}")
    checks["dataset_versions_nonempty"] = nonempty_table(dataset_versions_path)
    checks["annotation_audits_nonempty"] = audit_dir_has_outputs(audit_dir)
    checks["object_metrics_pass"] = object_metrics_pass(object_metrics_path, min_f1=min_f1)
    baseline_ok, baseline_notes = beats_majority_baseline(
        predictive_metrics_path, majority_baseline_rate=majority_baseline_rate, min_accuracy_lift=min_accuracy_lift
    )
    checks["beats_majority_baseline"] = baseline_ok
    notes.extend(baseline_notes)
    brier_ok, brier_notes = brier_within_threshold(predictive_metrics_path, max_brier=max_brier)
    checks["brier_within_threshold"] = brier_ok
    notes.extend(brier_notes)
    for name, ok in checks.items():
        if not ok and name not in {"cards_valid"}:
            notes.append(f"check failed: {name}")
    return PromotionGateResult(ok=all(checks.values()), checks=checks, notes=notes)


def write_promotion_gate_report(
    output: Path,
    model_card_path: Path | None,
    data_card_path: Path | None,
    dataset_versions_path: Path,
    audit_dir: Path,
    object_metrics_path: Path,
    min_f1: float = 0.0,
    predictive_metrics_path: Path | None = None,
    majority_baseline_rate: float = 0.0,
    min_accuracy_lift: float = 0.0,
    max_brier: float | None = None,
) -> Path:
    result = run_promotion_gate(
        model_card_path=model_card_path,
        data_card_path=data_card_path,
        dataset_versions_path=dataset_versions_path,
        audit_dir=audit_dir,
        object_metrics_path=object_metrics_path,
        min_f1=min_f1,
        predictive_metrics_path=predictive_metrics_path,
        majority_baseline_rate=majority_baseline_rate,
        min_accuracy_lift=min_accuracy_lift,
        max_brier=max_brier,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Promotion Gate", "", f"ok: {str(result.ok).lower()}", "", "## Checks"]
    lines.extend(f"- {name}: {str(ok).lower()}" for name, ok in result.checks.items())
    lines.extend(["", "## Notes"])
    lines.extend(f"- {note}" for note in result.notes) if result.notes else lines.append("- No notes.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def promotion_gate_dict(result: PromotionGateResult) -> dict:
    return asdict(result)
