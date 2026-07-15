import pandas as pd

from soccer_edge.promotion_gate import nonempty_table, object_metrics_pass, run_promotion_gate, write_promotion_gate_report


def test_nonempty_table(tmp_path) -> None:
    path = tmp_path / "rows.csv"
    pd.DataFrame([{"x": 1}]).to_csv(path, index=False)
    assert nonempty_table(path)


def test_object_metrics_pass(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    pd.DataFrame([{"class_name": "player", "f1": 0.8}]).to_csv(path, index=False)
    assert object_metrics_pass(path, min_f1=0.5)
    assert not object_metrics_pass(path, min_f1=0.9)


def test_write_promotion_gate_report(tmp_path) -> None:
    model_card = tmp_path / "MODEL_CARD.md"
    data_card = tmp_path / "DATA_CARD.md"
    versions = tmp_path / "versions.csv"
    audit_dir = tmp_path / "audit"
    metrics = tmp_path / "metrics.csv"
    output = tmp_path / "promotion.md"
    audit_dir.mkdir()
    model_card.write_text("# Model Card\n\n## Intended use\n\n## Features\n\n## Metrics\n\n## Limitations\n", encoding="utf-8")
    data_card.write_text("# Data Card\n\nRights status: compatible_license\n\n## Sources\n\n## Lineage\n\n## Allowed use\n\n## Restrictions\n", encoding="utf-8")
    pd.DataFrame([{"path": "x", "sha256": "abc"}]).to_csv(versions, index=False)
    pd.DataFrame([{"class_name": "player", "row_count": 1}]).to_csv(audit_dir / "by_class.csv", index=False)
    pd.DataFrame([{"class_name": "player", "f1": 1.0}]).to_csv(metrics, index=False)
    result = run_promotion_gate(model_card, data_card, versions, audit_dir, metrics)
    assert result.ok
    path = write_promotion_gate_report(output, model_card, data_card, versions, audit_dir, metrics)
    assert path.exists()


def test_beats_majority_baseline(tmp_path) -> None:
    from soccer_edge.promotion_gate import beats_majority_baseline

    metrics = tmp_path / "pred.csv"
    pd.DataFrame([{"accuracy": 0.55, "brier": 0.30}]).to_csv(metrics, index=False)
    ok, notes = beats_majority_baseline(metrics, majority_baseline_rate=0.50, min_accuracy_lift=0.02)
    assert ok, notes
    ok_bad, _ = beats_majority_baseline(metrics, majority_baseline_rate=0.60, min_accuracy_lift=0.0)
    assert not ok_bad
    ok_skip, _ = beats_majority_baseline(None)
    assert ok_skip


def test_beats_majority_baseline_uses_recorded_baseline(tmp_path) -> None:
    """When no rate is passed, the gate must use the recorded baseline_accuracy."""
    from soccer_edge.promotion_gate import beats_majority_baseline

    metrics = tmp_path / "pred.csv"
    # No-lift model: accuracy equals the recorded baseline. Must FAIL with default lift.
    pd.DataFrame([{"accuracy": 0.50, "brier": 0.30, "baseline_accuracy": 0.50}]).to_csv(metrics, index=False)
    ok, notes = beats_majority_baseline(metrics, min_accuracy_lift=0.02)
    assert not ok, notes
    # Real lift over the recorded baseline passes.
    good = tmp_path / "good.csv"
    pd.DataFrame([{"accuracy": 0.60, "brier": 0.30, "baseline_accuracy": 0.50}]).to_csv(good, index=False)
    ok_good, _ = beats_majority_baseline(good, min_accuracy_lift=0.02)
    assert ok_good


def test_beats_majority_baseline_fallback_note(tmp_path) -> None:
    from soccer_edge.promotion_gate import beats_majority_baseline

    metrics = tmp_path / "pred.csv"
    pd.DataFrame([{"accuracy": 0.10, "brier": 0.30}]).to_csv(metrics, index=False)
    ok, notes = beats_majority_baseline(metrics, min_accuracy_lift=0.02)
    assert ok  # 0.10 >= 0.0 + 0.02
    assert any("using 0.0" in note for note in notes)


def test_brier_within_threshold(tmp_path) -> None:
    from soccer_edge.promotion_gate import brier_within_threshold

    metrics = tmp_path / "pred.csv"
    pd.DataFrame([{"accuracy": 0.55, "brier": 0.30}]).to_csv(metrics, index=False)
    ok, notes = brier_within_threshold(metrics, max_brier=0.40)
    assert ok, notes
    ok_bad, _ = brier_within_threshold(metrics, max_brier=0.20)
    assert not ok_bad


def test_promotion_gate_blocks_no_lift(tmp_path) -> None:
    model_card = tmp_path / "MODEL_CARD.md"
    data_card = tmp_path / "DATA_CARD.md"
    versions = tmp_path / "versions.csv"
    audit_dir = tmp_path / "audit"
    object_metrics = tmp_path / "metrics.csv"
    pred_metrics = tmp_path / "pred.csv"
    audit_dir.mkdir()
    model_card.write_text("# Model Card\n\n## Intended use\n\n## Features\n\n## Metrics\n\n## Limitations\n", encoding="utf-8")
    data_card.write_text("# Data Card\n\nRights status: compatible_license\n\n## Sources\n\n## Lineage\n\n## Allowed use\n\n## Restrictions\n", encoding="utf-8")
    pd.DataFrame([{"path": "x", "sha256": "abc"}]).to_csv(versions, index=False)
    pd.DataFrame([{"class_name": "player", "row_count": 1}]).to_csv(audit_dir / "by_class.csv", index=False)
    pd.DataFrame([{"class_name": "player", "f1": 1.0}]).to_csv(object_metrics, index=False)
    # CNN highlight winner shows no lift at ~0.50 accuracy / ~0.62 brier.
    pd.DataFrame([{"accuracy": 0.50, "brier": 0.62}]).to_csv(pred_metrics, index=False)
    result = run_promotion_gate(
        model_card,
        data_card,
        versions,
        audit_dir,
        object_metrics,
        predictive_metrics_path=pred_metrics,
        majority_baseline_rate=0.50,
        min_accuracy_lift=0.02,
        max_brier=0.50,
    )
    assert not result.ok
    assert result.checks["beats_majority_baseline"] is False
    assert result.checks["brier_within_threshold"] is False
