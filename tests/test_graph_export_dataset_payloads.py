from soccer_edge.store.graph_export import annotation_audit_payload, dataset_version_payload, object_evaluation_payload


def test_dataset_version_payload() -> None:
    payload = dataset_version_payload({"path": "a.csv", "sha256": "abc", "size_bytes": 10})
    assert "DatasetVersion" in payload["statement"]
    assert payload["props"]["version_id"] == "a.csv::abc"


def test_annotation_audit_payload() -> None:
    payload = annotation_audit_payload({"audit_name": "by_class", "class_name": "player", "row_count": 4})
    assert "AnnotationAudit" in payload["statement"]
    assert "player" in payload["props"]["audit_id"]


def test_object_evaluation_payload() -> None:
    payload = object_evaluation_payload({"class_name": "ball", "precision": 1.0, "recall": 0.5, "f1": 0.67})
    assert "ObjectEvaluation" in payload["statement"]
    assert payload["props"]["evaluation_id"].startswith("ball::")
