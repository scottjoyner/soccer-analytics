from pathlib import Path

import pandas as pd

from soccer_edge.media_reader import require_media_reader
from soccer_edge.media_samples import iter_media_samples


def frame_image_name(frame_idx: int, prefix: str = "frame", extension: str = ".jpg") -> str:
    clean_extension = extension if extension.startswith(".") else f".{extension}"
    return f"{prefix}_{frame_idx:08d}{clean_extension}"


def export_video_frames(
    input_path: Path,
    output_dir: Path,
    stride: int = 1,
    max_frames: int | None = None,
    prefix: str = "frame",
    extension: str = ".jpg",
) -> pd.DataFrame:
    reader = require_media_reader()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for sample in iter_media_samples(input_path, stride=stride, max_samples=max_frames):
        image_path = output_dir / frame_image_name(sample.index, prefix=prefix, extension=extension)
        reader.imwrite(str(image_path), sample.data)
        rows.append({"frame_idx": sample.index, "timestamp_seconds": sample.time_seconds, "image_path": str(image_path)})
    return pd.DataFrame(rows)


def export_video_frame_manifest(
    input_path: Path,
    output_dir: Path,
    manifest_output: Path,
    stride: int = 1,
    max_frames: int | None = None,
    prefix: str = "frame",
    extension: str = ".jpg",
) -> Path:
    manifest = export_video_frames(input_path, output_dir, stride=stride, max_frames=max_frames, prefix=prefix, extension=extension)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_output, index=False)
    return manifest_output
