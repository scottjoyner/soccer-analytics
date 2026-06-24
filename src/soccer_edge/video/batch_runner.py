from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from soccer_edge.video.manifest import VideoManifestRow, read_video_manifest, validate_processable_video


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
