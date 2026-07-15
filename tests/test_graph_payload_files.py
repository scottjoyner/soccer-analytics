import json

import pandas as pd
import pytest

from soccer_edge.graph_payload_files import write_annotation_audit_payloads, write_graph_payloads


def test_write_graph_payloads_dataset_version(tmp_path) -> None:
    source = tmp_path / "versions.csv"
    output = tmp_path / "payloads.jsonl"
    pd.DataFrame([{"path": "a.csv", "sha256": "abc", "size_bytes": 1}]).to_csv(source, index=False)
    path = write_graph_payloads(source, output, "dataset-version")
    assert path.exists()
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "DatasetVersion" in row["statement"]


def test_write_annotation_audit_payloads(tmp_path) -> None:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    pd.DataFrame([{"class_name": "player", "row_count": 2}]).to_csv(audit_dir / "by_class.csv", index=False)
    output = tmp_path / "audit.jsonl"
    path = write_annotation_audit_payloads(audit_dir, output)
    assert path.exists()
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["props"]["audit_name"] == "by_class"


def test_write_graph_payloads_validates_kind(tmp_path) -> None:
    source = tmp_path / "rows.csv"
    pd.DataFrame([{"x": 1}]).to_csv(source, index=False)
    with pytest.raises(ValueError):
        write_graph_payloads(source, tmp_path / "out.jsonl", "bad-kind")


def test_write_graph_payloads_player_match(tmp_path) -> None:
    source = tmp_path / "player_match.csv"
    output = tmp_path / "payloads.jsonl"
    pd.DataFrame([{"match_id": "m1", "player_name": "A", "goals": 2}]).to_csv(source, index=False)
    path = write_graph_payloads(source, output, "player-match")
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "PlayerMatch" in row["statement"]
    assert row["props"]["player_match_id"] == "m1::A"


def test_write_graph_payloads_player_form(tmp_path) -> None:
    source = tmp_path / "player_form.csv"
    output = tmp_path / "payloads.jsonl"
    pd.DataFrame([{"player_name": "A", "match_id": "m1", "goals_form_5": 0.3}]).to_csv(source, index=False)
    path = write_graph_payloads(source, output, "player-form")
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert "PlayerForm" in row["statement"]
    assert row["props"]["player_form_id"] == "A::m1"
