from pathlib import Path

import pandas as pd

from soccer_edge.dataset_versioning import dataset_version_id, dataset_versions
from soccer_edge.training_sources import training_sources_frame


def table_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in frame.iterrows()]
    return "\n".join([header, divider, *rows])


def manifest_stats(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    stats: dict[str, object] = {"path": str(path), "rows": len(frame), "columns": len(frame.columns)}
    if "rights_status" in frame.columns:
        stats["rights_statuses"] = ",".join(str(value) for value in sorted(frame["rights_status"].dropna().unique()))
    if "class_name" in frame.columns:
        stats["classes"] = ",".join(str(value) for value in sorted(frame["class_name"].dropna().unique()))
    return stats


def auto_data_card_markdown(
    dataset_name: str,
    manifests: list[Path],
    rights_status: str = "owned",
    version_paths: list[Path] | None = None,
) -> str:
    manifest_frame = pd.DataFrame([manifest_stats(path) for path in manifests]) if manifests else pd.DataFrame()
    sources = training_sources_frame()
    selected_version_paths = version_paths or manifests
    versions = dataset_versions(selected_version_paths) if selected_version_paths else pd.DataFrame()
    version_id = dataset_version_id(selected_version_paths) if selected_version_paths else "unknown"
    return "\n".join(
        [
            f"# Data Card: {dataset_name}",
            "",
            f"Rights status: {rights_status}",
            f"Dataset version ID: {version_id}",
            "",
            "## Source catalog",
            table_markdown(sources[["name", "tier", "modality", "rights_posture"]]),
            "",
            "## Manifest stats",
            table_markdown(manifest_frame),
            "",
            "## Asset versions",
            table_markdown(versions),
            "",
            "## Allowed use",
            "Offline research, calibration, training, and evaluation within this repository.",
            "",
            "## Restrictions",
            "Do not process or redistribute footage without owned, licensed, or compatible-license rights status.",
            "",
        ]
    )


def write_auto_data_card(
    dataset_name: str,
    manifests: list[Path],
    output: Path,
    rights_status: str = "owned",
    version_paths: list[Path] | None = None,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(auto_data_card_markdown(dataset_name, manifests, rights_status=rights_status, version_paths=version_paths), encoding="utf-8")
    return output
