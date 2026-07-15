from dataclasses import asdict, is_dataclass
from typing import Any


def row_to_properties(row: object) -> dict[str, Any]:
    if is_dataclass(row):
        return asdict(row)
    if isinstance(row, dict):
        return dict(row)
    raise TypeError("row must be a dataclass or dict")


def merge_node_statement(label: str, key: str = "id") -> str:
    return f"MERGE (n:{label} {{{key}: $props.{key}}}) SET n += $props"


def build_node_payload(label: str, row: object, key: str = "id") -> dict[str, Any]:
    props = row_to_properties(row)
    if key not in props:
        raise ValueError(f"missing key property: {key}")
    return {"statement": merge_node_statement(label, key), "props": props}


def ensure_id(row: object, id_field: str, fallback_fields: list[str]) -> dict[str, Any]:
    props = row_to_properties(row)
    if id_field not in props:
        values = [str(props.get(field, "")) for field in fallback_fields]
        props[id_field] = "::".join(values)
    return props


def match_payload(row: object) -> dict[str, Any]:
    return build_node_payload("Match", row, key="match_id")


def frame_payload(row: object) -> dict[str, Any]:
    return build_node_payload("Frame", row, key="frame_idx")


def feature_payload(row: object) -> dict[str, Any]:
    return build_node_payload("Feature", row, key="feature_name")


def run_payload(row: object) -> dict[str, Any]:
    return build_node_payload("ModelRun", row, key="name")


def dataset_version_payload(row: object) -> dict[str, Any]:
    props = ensure_id(row, "version_id", ["path", "sha256"])
    return build_node_payload("DatasetVersion", props, key="version_id")


def annotation_audit_payload(row: object) -> dict[str, Any]:
    props = ensure_id(row, "audit_id", ["audit_name", "class_name", "frame_idx", "split"])
    return build_node_payload("AnnotationAudit", props, key="audit_id")


def object_evaluation_payload(row: object) -> dict[str, Any]:
    props = ensure_id(row, "evaluation_id", ["class_name", "precision", "recall", "f1"])
    return build_node_payload("ObjectEvaluation", props, key="evaluation_id")


def player_match_payload(row: object) -> dict[str, Any]:
    props = ensure_id(row, "player_match_id", ["match_id", "player_name"])
    return build_node_payload("PlayerMatch", props, key="player_match_id")


def player_form_payload(row: object) -> dict[str, Any]:
    props = ensure_id(row, "player_form_id", ["player_name", "match_id"])
    return build_node_payload("PlayerForm", props, key="player_form_id")
