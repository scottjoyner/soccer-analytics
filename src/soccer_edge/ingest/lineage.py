from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def lineage_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_lineage_columns(
    frame: pd.DataFrame,
    source_name: str,
    source_path: Path | str,
    dataset_version: str = "unknown",
) -> pd.DataFrame:
    output = frame.copy()
    output["lineage_source_name"] = source_name
    output["lineage_source_path"] = str(source_path)
    output["lineage_dataset_version"] = dataset_version
    output["lineage_ingested_at_utc"] = lineage_timestamp()
    return output


def add_lineage_to_tables(
    tables: dict[str, pd.DataFrame],
    source_name: str,
    source_path: Path | str,
    dataset_version: str = "unknown",
) -> dict[str, pd.DataFrame]:
    return {
        name: add_lineage_columns(frame, source_name=source_name, source_path=source_path, dataset_version=dataset_version)
        for name, frame in tables.items()
    }
