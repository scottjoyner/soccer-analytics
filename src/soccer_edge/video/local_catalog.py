from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
PROCESSABLE_RIGHTS = {"owned", "licensed", "compatible_license"}


@dataclass(frozen=True)
class LocalVideoCatalogRow:
    video_id: str
    match_id: str
    clip_type: str
    local_path: str
    rights_status: str
    rights_reference: str = ""
    notes: str = ""


def discover_local_videos(
    root: Path,
    rights_status: str = "owned",
    clip_type: str = "full_match",
    rights_reference: str = "",
) -> list[LocalVideoCatalogRow]:
    if rights_status not in PROCESSABLE_RIGHTS:
        raise ValueError(f"rights_status must be one of {sorted(PROCESSABLE_RIGHTS)}")
    if rights_status in PROCESSABLE_RIGHTS and not rights_reference:
        raise ValueError(
            f"rights_status={rights_status!r} requires a recorded rights_reference "
            "(explicit written rights) before the footage may be used."
        )
    rows: list[LocalVideoCatalogRow] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            video_id = path.stem.replace(" ", "_")
            rows.append(
                LocalVideoCatalogRow(
                    video_id=video_id,
                    match_id=video_id,
                    clip_type=clip_type,
                    local_path=str(path),
                    rights_status=rights_status,
                    rights_reference=rights_reference,
                    notes="local approved footage",
                )
            )
    return rows


def write_local_video_catalog(
    root: Path,
    output: Path,
    rights_status: str = "owned",
    clip_type: str = "full_match",
    rights_reference: str = "",
) -> Path:
    rows = discover_local_videos(
        root=root, rights_status=rights_status, clip_type=clip_type, rights_reference=rights_reference
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(row) for row in rows]).to_csv(output, index=False)
    return output
