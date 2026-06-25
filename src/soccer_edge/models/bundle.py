from pathlib import Path
from typing import Any

import joblib

from soccer_edge.models.run_metadata import RunMetadata, read_run_metadata, utc_now_iso, write_run_metadata


def save_bundle(
    model: Any,
    output_dir: Path,
    name: str,
    version: str,
    feature_names: list[str],
    metrics: dict[str, float],
    notes: str = "",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    object_path = output_dir / "model.joblib"
    info_path = output_dir / "metadata.json"
    joblib.dump(model, object_path)
    write_run_metadata(
        RunMetadata(
            name=name,
            version=version,
            created_at_utc=utc_now_iso(),
            feature_names=feature_names,
            metrics=metrics,
            notes=notes,
        ),
        info_path,
    )
    return {"model": object_path, "metadata": info_path}


def load_bundle(bundle_dir: Path) -> tuple[Any, RunMetadata]:
    return joblib.load(bundle_dir / "model.joblib"), read_run_metadata(bundle_dir / "metadata.json")
