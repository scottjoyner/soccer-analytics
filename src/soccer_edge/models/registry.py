from pathlib import Path

import pandas as pd

from soccer_edge.models.run_metadata import read_run_metadata


def registry_row(bundle_dir: Path) -> dict[str, object]:
    metadata = read_run_metadata(bundle_dir / "metadata.json")
    return {
        "name": metadata.name,
        "version": metadata.version,
        "created_at_utc": metadata.created_at_utc,
        "feature_count": len(metadata.feature_names),
        "metrics": metadata.metrics,
        "bundle_dir": str(bundle_dir),
    }


def build_registry_index(root_dir: Path) -> pd.DataFrame:
    rows = []
    for metadata_path in sorted(root_dir.rglob("metadata.json")):
        rows.append(registry_row(metadata_path.parent))
    return pd.DataFrame(rows)


def write_registry_index(root_dir: Path, output_path: Path) -> Path:
    frame = build_registry_index(root_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        frame.to_parquet(output_path, index=False)
    else:
        frame.to_csv(output_path, index=False)
    return output_path
