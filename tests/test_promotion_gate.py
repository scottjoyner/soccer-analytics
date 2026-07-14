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
    data_card.write_text("# Data Card\n\n## Sources\n\n## Lineage\n\n## Allowed use\n\n## Restrictions\n", encoding="utf-8")
    pd.DataFrame([{"path": "x", "sha256": "abc"}]).to_csv(versions, index=False)
    pd.DataFrame([{"class_name": "player", "row_count": 1}]).to_csv(audit_dir / "by_class.csv", index=False)
    pd.DataFrame([{"class_name": "player", "f1": 1.0}]).to_csv(metrics, index=False)
    result = run_promotion_gate(model_card, data_card, versions, audit_dir, metrics)
    assert result.ok
    path = write_promotion_gate_report(output, model_card, data_card, versions, audit_dir, metrics)
    assert path.exists()
