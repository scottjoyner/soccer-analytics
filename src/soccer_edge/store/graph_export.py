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


def match_payload(row: object) -> dict[str, Any]:
    return build_node_payload("Match", row, key="match_id")


def frame_payload(row: object) -> dict[str, Any]:
    return build_node_payload("Frame", row, key="frame_idx")


def feature_payload(row: object) -> dict[str, Any]:
    return build_node_payload("Feature", row, key="feature_name")


def run_payload(row: object) -> dict[str, Any]:
    return build_node_payload("ModelRun", row, key="name")
