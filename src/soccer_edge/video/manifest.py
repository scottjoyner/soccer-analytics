from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from soccer_edge.video.modality_rules import ModalityRules

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


MANIFEST_FIELDS = [
    "video_id",
    "match_id",
    "competition",
    "season",
    "home_team",
    "away_team",
    "clip_type",
    "source_url",
    "local_path",
    "period",
    "start_match_second",
    "end_match_second",
    "rights_status",
    "rights_reference",
    "notes",
]


def manifest_row_to_dict(row: VideoManifestRow) -> dict[str, str]:
    return {
        "video_id": row.video_id,
        "match_id": row.match_id,
        "competition": row.competition,
        "season": row.season,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "clip_type": row.clip_type,
        "source_url": row.source_url,
        "local_path": str(row.local_path),
        "period": row.period,
        "start_match_second": "" if row.start_match_second is None else str(row.start_match_second),
        "end_match_second": "" if row.end_match_second is None else str(row.end_match_second),
        "rights_status": row.rights_status,
        "rights_reference": row.rights_reference,
        "notes": row.notes,
    }


def append_manifest_row(manifest_path: Path, row: VideoManifestRow) -> Path:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(manifest_row_to_dict(row))
    return manifest_path


def find_manifest_row(manifest_path: Path, video_id: str) -> VideoManifestRow | None:
    for row in read_video_manifest(manifest_path):
        if row.video_id == video_id:
            return row
    return None


def validate_processable_video(
    row: VideoManifestRow,
    licensed_root: Path,
    modality_rules: ModalityRules | None = None,
) -> None:
    if not row.is_processable:
        raise ValueError(f"Video {row.video_id} is not processable: rights_status={row.rights_status}")

    if not row.rights_reference:
        raise ValueError(
            f"Video {row.video_id} is missing a recorded rights_reference; "
            "explicit written rights must be recorded before processing."
        )

    rules = modality_rules or ModalityRules.default()
    blocked = rules.blocked_reason(row)
    if blocked is not None:
        raise ValueError(
            f"Video {row.video_id} uses blocked modality {blocked!r}: "
            "public/remote sources are discovery metadata only and cannot be processed."
        )

    root = licensed_root.resolve()
    local_path = row.local_path.resolve()
    if root not in local_path.parents and local_path != root:
        raise ValueError(f"Video {row.video_id} is outside licensed root: {local_path}")

    if not local_path.exists():
        raise ValueError(
            f"Video {row.video_id} local_path does not exist: {local_path}; "
            "cannot process a manifest row whose file is missing."
        )

    # Resolve through any symlinks so a symlink inside licensed_root that points
    # elsewhere cannot smuggle in an unapproved file, and refuse non-regular files.
    real_path = local_path.resolve()
    if real_path.is_symlink() or real_path != local_path:
        if root not in real_path.parents and real_path != root:
            raise ValueError(
                f"Video {row.video_id} resolves outside licensed root via symlink: {real_path}"
            )
    if not real_path.is_file():
        raise ValueError(f"Video {row.video_id} local_path is not a regular file: {real_path}")
