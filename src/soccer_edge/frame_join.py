from pathlib import Path

import pandas as pd


def attach_image_paths(
    detections: pd.DataFrame,
    frame_manifest: pd.DataFrame,
    frame_column: str = "frame_idx",
    image_path_column: str = "image_path",
) -> pd.DataFrame:
    if frame_column not in detections.columns:
        raise ValueError(f"detections missing frame column: {frame_column}")
    if frame_column not in frame_manifest.columns:
        raise ValueError(f"frame manifest missing frame column: {frame_column}")
    if image_path_column not in frame_manifest.columns:
        raise ValueError(f"frame manifest missing image path column: {image_path_column}")
    manifest = frame_manifest[[frame_column, image_path_column]].drop_duplicates(frame_column)
    return detections.merge(manifest, on=frame_column, how="left")


def attach_image_paths_from_tables(
    detections_path: Path,
    frame_manifest_path: Path,
    output_path: Path,
    frame_column: str = "frame_idx",
    image_path_column: str = "image_path",
) -> Path:
    detections = pd.read_parquet(detections_path) if detections_path.suffix == ".parquet" else pd.read_csv(detections_path)
    manifest = pd.read_parquet(frame_manifest_path) if frame_manifest_path.suffix == ".parquet" else pd.read_csv(frame_manifest_path)
    joined = attach_image_paths(detections, manifest, frame_column=frame_column, image_path_column=image_path_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        joined.to_parquet(output_path, index=False)
    else:
        joined.to_csv(output_path, index=False)
    return output_path
