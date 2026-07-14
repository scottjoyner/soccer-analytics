from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from soccer_edge.video.manifest import VideoManifestRow, find_manifest_row, read_video_manifest, validate_processable_video


@dataclass(frozen=True)
class VideoProcessingPlan:
    processable: list[VideoManifestRow]
    skipped: list[VideoManifestRow]


def build_processing_plan(manifest_path: Path, licensed_root: Path) -> VideoProcessingPlan:
    rows = read_video_manifest(manifest_path)
    processable: list[VideoManifestRow] = []
    skipped: list[VideoManifestRow] = []

    for row in rows:
        try:
            validate_processable_video(row, licensed_root)
        except ValueError:
            skipped.append(row)
        else:
            processable.append(row)

    return VideoProcessingPlan(processable=processable, skipped=skipped)


def assert_processable(
    manifest_path: Path,
    video_id: str,
    input_path: Path,
    licensed_root: Path,
) -> VideoManifestRow:
    """Fail fast unless ``video_id`` is an approved, rights-referenced row whose
    local_path matches ``input_path``. Used as a defense-in-depth gate before any
    footage is opened for inference."""

    row = find_manifest_row(manifest_path, video_id)
    if row is None:
        raise ValueError(f"No manifest row for video_id={video_id!r}; record footage rights before processing.")

    validate_processable_video(row, licensed_root)

    if Path(input_path).resolve() != row.local_path.resolve():
        raise ValueError(
            f"Input {input_path} does not match the approved local_path "
            f"{row.local_path} for video_id={video_id!r}."
        )
    return row
