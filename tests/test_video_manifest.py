from pathlib import Path

import pytest

from soccer_edge.video.manifest import manifest_row_from_dict, validate_processable_video


def _licensed_file(rel: str) -> Path:
    path = Path("data/raw/video_licensed") / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-video")
    return path


def test_manifest_row_pending_is_not_processable() -> None:
    row = manifest_row_from_dict(
        {
            "video_id": "clip_1",
            "match_id": "match_1",
            "clip_type": "goal_montage",
            "local_path": "data/raw/video_licensed/world_cup/goal_montages/clip_1.mp4",
            "rights_status": "pending",
        }
    )
    assert row.is_processable is False


def test_manifest_row_licensed_inside_root_is_processable() -> None:
    path = _licensed_file("world_cup/full_matches/clip_2.mp4")
    row = manifest_row_from_dict(
        {
            "video_id": "clip_2",
            "match_id": "match_2",
            "clip_type": "full_match",
            "local_path": str(path),
            "rights_status": "licensed",
            "rights_reference": "license-file:///perms/wc2026.pdf",
        }
    )
    validate_processable_video(row, Path("data/raw/video_licensed"))


def test_manifest_row_processable_without_rights_reference_is_rejected() -> None:
    path = _licensed_file("world_cup/full_matches/clip_2b.mp4")
    row = manifest_row_from_dict(
        {
            "video_id": "clip_2b",
            "match_id": "match_2b",
            "clip_type": "full_match",
            "local_path": str(path),
            "rights_status": "licensed",
        }
    )
    with pytest.raises(ValueError):
        validate_processable_video(row, Path("data/raw/video_licensed"))


def test_manifest_row_outside_root_is_rejected() -> None:
    row = manifest_row_from_dict(
        {
            "video_id": "clip_3",
            "match_id": "match_3",
            "clip_type": "highlight",
            "local_path": "data/raw/not_licensed/clip_3.mp4",
            "rights_status": "owned",
        }
    )
    with pytest.raises(ValueError):
        validate_processable_video(row, Path("data/raw/video_licensed"))


def test_manifest_row_missing_file_is_rejected() -> None:
    row = manifest_row_from_dict(
        {
            "video_id": "clip_4",
            "match_id": "match_4",
            "clip_type": "highlight",
            "local_path": "data/raw/video_licensed/missing/clip_4.mp4",
            "rights_status": "owned",
            "rights_reference": "license-file:///perms/x.pdf",
        }
    )
    with pytest.raises(ValueError, match="does not exist"):
        validate_processable_video(row, Path("data/raw/video_licensed"))
