from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

PROCESSABLE_RIGHTS_STATUSES = {"owned", "licensed", "compatible_license"}
VALID_CLIP_TYPES = {"full_match", "goal_montage", "highlight", "training_clip", "unknown"}


@dataclass(frozen=True)
class VideoManifestRow:
    video_id: str
    match_id: str
    competition: str
    season: str
    home_team: str
    away_team: str
    clip_type: str
    source_url: str
    local_path: Path
    period: str
    start_match_second: float | None
    end_match_second: float | None
    rights_status: str
    rights_reference: str = ""
    notes: str = ""

    @property
    def is_processable(self) -> bool:
        return self.rights_status in PROCESSABLE_RIGHTS_STATUSES


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def read_video_manifest(path: Path) -> list[VideoManifestRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [manifest_row_from_dict(row) for row in reader]


def manifest_row_from_dict(row: dict[str, str]) -> VideoManifestRow:
    clip_type = row.get("clip_type", "unknown")
    if clip_type not in VALID_CLIP_TYPES:
        raise ValueError(f"Invalid clip_type: {clip_type}")

    return VideoManifestRow(
        video_id=row["video_id"],
        match_id=row["match_id"],
        competition=row.get("competition", ""),
        season=row.get("season", ""),
        home_team=row.get("home_team", ""),
        away_team=row.get("away_team", ""),
        clip_type=clip_type,
        source_url=row.get("source_url", ""),
        local_path=Path(row["local_path"]),
        period=row.get("period", ""),
        start_match_second=parse_optional_float(row.get("start_match_second")),
        end_match_second=parse_optional_float(row.get("end_match_second")),
        rights_status=row.get("rights_status", "pending"),
        rights_reference=row.get("rights_reference", ""),
        notes=row.get("notes", ""),
    )


def find_manifest_row(manifest_path: Path, video_id: str) -> VideoManifestRow | None:
    for row in read_video_manifest(manifest_path):
        if row.video_id == video_id:
            return row
    return None


def validate_processable_video(row: VideoManifestRow, licensed_root: Path) -> None:
    if not row.is_processable:
        raise ValueError(f"Video {row.video_id} is not processable: rights_status={row.rights_status}")

    if not row.rights_reference:
        raise ValueError(
            f"Video {row.video_id} is missing a recorded rights_reference; "
            "explicit written rights must be recorded before processing."
        )

    root = licensed_root.resolve()
    local_path = row.local_path.resolve()
    if root not in local_path.parents and local_path != root:
        raise ValueError(f"Video {row.video_id} is outside licensed root: {local_path}")
