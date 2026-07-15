"""Promote a candidate model bundle once it passes the promotion gate.

`promote_bundle` runs the promotion gate against a candidate bundle's cards,
versions, audits, object metrics, and predictive metrics. If the gate passes it
copies the bundle plus those artifacts into a promoted location and writes a
`promotion.json` record, so the registry command can still index it.
"""

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from soccer_edge.models.run_metadata import read_run_metadata, utc_now_iso
from soccer_edge.promotion_gate import run_promotion_gate, write_promotion_gate_report


@dataclass(frozen=True)
class PromotionRecord:
    name: str
    version: str
    promoted_at_utc: str
    source_bundle: str
    gate_ok: bool
    accuracy: float | None = None
    brier: float | None = None
    majority_baseline_rate: float | None = None
    min_accuracy_lift: float = 0.0
    max_brier: float | None = None


def _read_metrics_values(predictive_metrics_path: Path | None) -> tuple[float | None, float | None]:
    if predictive_metrics_path is None or not Path(predictive_metrics_path).exists():
        return None, None
    frame = pd.read_csv(predictive_metrics_path)
    accuracy = float(frame["accuracy"].mean()) if "accuracy" in frame.columns else None
    brier = float(frame["brier"].mean()) if "brier" in frame.columns else None
    return accuracy, brier


def promote_bundle(
    bundle_dir: Path,
    promoted_root: Path,
    model_card_path: Path | None,
    data_card_path: Path | None,
    dataset_versions_path: Path,
    audit_dir: Path,
    object_metrics_path: Path,
    predictive_metrics_path: Path | None = None,
    majority_baseline_rate: float | None = None,
    min_accuracy_lift: float = 0.0,
    max_brier: float | None = None,
    min_f1: float = 0.0,
) -> Path:
    """Promote ``bundle_dir`` if the promotion gate passes.

    Raises ``RuntimeError`` when the gate fails (so callers can fail loudly) and
    ``FileExistsError`` when the promoted destination already exists.
    """

    bundle_dir = Path(bundle_dir)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"bundle_dir does not exist: {bundle_dir}")
    if not (bundle_dir / "metadata.json").exists():
        raise FileNotFoundError(f"bundle metadata.json missing: {bundle_dir / 'metadata.json'}")
    for label, path in (
        ("dataset_versions_path", dataset_versions_path),
        ("audit_dir", audit_dir),
        ("object_metrics_path", object_metrics_path),
    ):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"{label} does not exist: {p}")
    if model_card_path is not None and not Path(model_card_path).exists():
        raise FileNotFoundError(f"model_card_path does not exist: {model_card_path}")
    if data_card_path is not None and not Path(data_card_path).exists():
        raise FileNotFoundError(f"data_card_path does not exist: {data_card_path}")
    if predictive_metrics_path is not None and not Path(predictive_metrics_path).exists():
        raise FileNotFoundError(f"predictive_metrics_path does not exist: {predictive_metrics_path}")

    metadata = read_run_metadata(bundle_dir / "metadata.json")
    result = run_promotion_gate(
        model_card_path,
        data_card_path,
        dataset_versions_path,
        audit_dir,
        object_metrics_path,
        min_f1=min_f1,
        predictive_metrics_path=predictive_metrics_path,
        majority_baseline_rate=majority_baseline_rate,
        min_accuracy_lift=min_accuracy_lift,
        max_brier=max_brier,
    )
    if not result.ok:
        raise RuntimeError(f"promotion gate failed for {metadata.name} {metadata.version}: {result.notes}")

    dest = Path(promoted_root) / metadata.name / metadata.version
    if dest.exists():
        raise FileExistsError(f"promoted bundle already exists: {dest}")
    shutil.copytree(bundle_dir, dest)

    if model_card_path is not None:
        shutil.copy(model_card_path, dest / "MODEL_CARD.md")
    if data_card_path is not None:
        shutil.copy(data_card_path, dest / "DATA_CARD.md")
    shutil.copy(dataset_versions_path, dest / "dataset_versions.csv")
    shutil.copy(object_metrics_path, dest / "object_metrics.csv")
    if predictive_metrics_path is not None:
        shutil.copy(predictive_metrics_path, dest / "predictive_metrics.csv")
    write_promotion_gate_report(
        dest / "promotion_gate.md",
        model_card_path,
        data_card_path,
        dataset_versions_path,
        audit_dir,
        object_metrics_path,
        min_f1=min_f1,
        predictive_metrics_path=predictive_metrics_path,
        majority_baseline_rate=majority_baseline_rate,
        min_accuracy_lift=min_accuracy_lift,
        max_brier=max_brier,
    )

    accuracy, brier = _read_metrics_values(predictive_metrics_path)
    record = PromotionRecord(
        name=metadata.name,
        version=metadata.version,
        promoted_at_utc=utc_now_iso(),
        source_bundle=str(bundle_dir),
        gate_ok=True,
        accuracy=accuracy,
        brier=brier,
        majority_baseline_rate=majority_baseline_rate,
        min_accuracy_lift=min_accuracy_lift,
        max_brier=max_brier,
    )
    (dest / "promotion.json").write_text(json.dumps(asdict(record), indent=2, sort_keys=True), encoding="utf-8")
    return dest


def build_promoted_index(promoted_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(promoted_root).rglob("promotion.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def write_promoted_index(promoted_root: Path, output_path: Path) -> Path:
    frame = build_promoted_index(promoted_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        frame.to_parquet(output_path, index=False)
    else:
        frame.to_csv(output_path, index=False)
    return output_path
