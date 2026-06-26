from pathlib import Path

import pandas as pd

from soccer_edge.models.run_metadata import read_run_metadata


def registry_row(bundle_dir: Path) -> dict[str, object]:
    metadata = read_run_metadata(bundle_dir / "metadata.json")
    row: dict[str, object] = {
        "name": metadata.name,
        "version": metadata.version,
        "created_at_utc": metadata.created_at_utc,
        "feature_count": len(metadata.feature_names),
        "metrics": metadata.metrics,
        "bundle_dir": str(bundle_dir),
    }
    for metric_name, metric_value in metadata.metrics.items():
        row[f"metric_{metric_name}"] = metric_value
    return row


def build_registry_index(root_dir: Path) -> pd.DataFrame:
    rows = []
    for metadata_path in sorted(root_dir.rglob("metadata.json")):
        rows.append(registry_row(metadata_path.parent))
    return pd.DataFrame(rows)


def summarize_registry(frame: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    metric_column = f"metric_{metric}"
    sort_columns = [column for column in [metric_column, "created_at_utc"] if column in frame.columns]
    if not sort_columns:
        return frame.sort_values("created_at_utc", ascending=False) if "created_at_utc" in frame.columns else frame
    ascending = [False for _ in sort_columns]
    return frame.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def write_registry_index(root_dir: Path, output_path: Path) -> Path:
    frame = build_registry_index(root_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        frame.to_parquet(output_path, index=False)
    else:
        frame.to_csv(output_path, index=False)
    return output_path


def write_registry_summary(root_dir: Path, output_path: Path, metric: str = "accuracy") -> Path:
    frame = summarize_registry(build_registry_index(root_dir), metric=metric)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        frame.to_parquet(output_path, index=False)
    else:
        frame.to_csv(output_path, index=False)
    return output_path
