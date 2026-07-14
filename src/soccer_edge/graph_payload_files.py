import json
from pathlib import Path
from typing import Callable

import pandas as pd

from soccer_edge.store.graph_export import annotation_audit_payload, dataset_version_payload, object_evaluation_payload

PayloadBuilder = Callable[[dict], dict]


PAYLOAD_BUILDERS: dict[str, PayloadBuilder] = {
    "dataset-version": dataset_version_payload,
    "annotation-audit": annotation_audit_payload,
    "object-evaluation": object_evaluation_payload,
}


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def payload_rows(frame: pd.DataFrame, builder: PayloadBuilder, extra_props: dict | None = None) -> list[dict]:
    extra = extra_props or {}
    rows = []
    for row in frame.to_dict(orient="records"):
        row.update(extra)
        rows.append(builder(row))
    return rows


def write_graph_payloads(
    source: Path,
    output: Path,
    kind: str,
    extra_props: dict | None = None,
) -> Path:
    if kind not in PAYLOAD_BUILDERS:
        raise ValueError(f"unsupported graph payload kind: {kind}")
    frame = read_table(source)
    payloads = payload_rows(frame, PAYLOAD_BUILDERS[kind], extra_props=extra_props)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(payload, sort_keys=True) for payload in payloads) + ("\n" if payloads else ""), encoding="utf-8")
    return output


def write_annotation_audit_payloads(audit_dir: Path, output: Path) -> Path:
    payloads = []
    for path in sorted(audit_dir.glob("*.csv")):
        frame = read_table(path)
        payloads.extend(payload_rows(frame, annotation_audit_payload, extra_props={"audit_name": path.stem}))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(payload, sort_keys=True) for payload in payloads) + ("\n" if payloads else ""), encoding="utf-8")
    return output
