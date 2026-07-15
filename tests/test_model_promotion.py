from pathlib import Path

import pandas as pd

from soccer_edge.models.bundle import save_bundle
from soccer_edge.models.promotion import build_promoted_index, promote_bundle


def _cards(tmp_path: Path) -> tuple[Path, Path]:
    model_card = tmp_path / "MODEL_CARD.md"
    data_card = tmp_path / "DATA_CARD.md"
    model_card.write_text("# Model Card\n\n## Intended use\n\n## Features\n\n## Metrics\n\n## Limitations\n", encoding="utf-8")
    data_card.write_text("# Data Card\n\nRights status: compatible_license\n\n## Sources\n\n## Lineage\n\n## Allowed use\n\n## Restrictions\n", encoding="utf-8")
    return model_card, data_card


def _gate_inputs(tmp_path: Path, accuracy: float, brier: float) -> tuple[Path, Path, Path, Path]:
    versions = tmp_path / "versions.csv"
    audit_dir = tmp_path / "audit"
    object_metrics = tmp_path / "obj.csv"
    pred = tmp_path / "pred.csv"
    audit_dir.mkdir()
    pd.DataFrame([{"path": "x", "sha256": "abc"}]).to_csv(versions, index=False)
    pd.DataFrame([{"class_name": "player", "row_count": 1}]).to_csv(audit_dir / "by_class.csv", index=False)
    pd.DataFrame([{"class_name": "player", "f1": 1.0}]).to_csv(object_metrics, index=False)
    pd.DataFrame([{"model": "m", "accuracy": accuracy, "brier": brier, "baseline_accuracy": 0.5, "split": "test"}]).to_csv(pred, index=False)
    return versions, audit_dir, object_metrics, pred


def test_promote_bundle_passes(tmp_path) -> None:
    bundle = tmp_path / "candidate"
    save_bundle(model={"kind": "demo"}, output_dir=bundle, name="demo", version="v1", feature_names=["x"], metrics={"accuracy": 0.6})
    model_card, data_card = _cards(tmp_path)
    versions, audit_dir, object_metrics, pred = _gate_inputs(tmp_path, accuracy=0.60, brier=0.30)

    dest = promote_bundle(
        bundle_dir=bundle,
        promoted_root=tmp_path / "promoted",
        model_card_path=model_card,
        data_card_path=data_card,
        dataset_versions_path=versions,
        audit_dir=audit_dir,
        object_metrics_path=object_metrics,
        predictive_metrics_path=pred,
        majority_baseline_rate=0.50,
        min_accuracy_lift=0.02,
        max_brier=0.50,
    )
    assert dest.exists()
    assert (dest / "promotion.json").exists()
    assert (dest / "MODEL_CARD.md").exists()
    assert (dest / "promotion_gate.md").exists()
    assert (dest / "metadata.json").exists()
    frame = build_promoted_index(tmp_path / "promoted")
    assert len(frame) == 1
    assert frame.iloc[0]["name"] == "demo"
    assert frame.iloc[0]["accuracy"] == 0.60


def test_promote_bundle_rejects_no_lift(tmp_path) -> None:
    bundle = tmp_path / "candidate"
    save_bundle(model={"kind": "demo"}, output_dir=bundle, name="demo", version="v1", feature_names=["x"], metrics={"accuracy": 0.5})
    model_card, data_card = _cards(tmp_path)
    versions, audit_dir, object_metrics, pred = _gate_inputs(tmp_path, accuracy=0.50, brier=0.62)

    try:
        promote_bundle(
            bundle_dir=bundle,
            promoted_root=tmp_path / "promoted",
            model_card_path=model_card,
            data_card_path=data_card,
            dataset_versions_path=versions,
            audit_dir=audit_dir,
            object_metrics_path=object_metrics,
            predictive_metrics_path=pred,
            majority_baseline_rate=0.50,
            min_accuracy_lift=0.02,
            max_brier=0.50,
        )
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert not (tmp_path / "promoted").exists()
