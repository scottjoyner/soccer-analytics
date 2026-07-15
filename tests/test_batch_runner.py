from pathlib import Path

import pytest

from soccer_edge.video.batch_runner import assert_processable, build_processing_plan
from soccer_edge.video.manifest import VideoManifestRow, validate_processable_video


def _row(local_path: Path, source_url: str = "", clip_type: str = "full_match") -> VideoManifestRow:
    return VideoManifestRow(
        video_id="vid",
        match_id="match_1",
        competition="",
        season="",
        home_team="",
        away_team="",
        clip_type=clip_type,
        source_url=source_url,
        local_path=local_path,
        period="full",
        start_match_second=0.0,
        end_match_second=3.0,
        rights_status="licensed",
        rights_reference="license-file:///perms/wc2026.pdf",
    )


def test_build_processing_plan_filters_unapproved_rows(tmp_path: Path) -> None:
    licensed_root = tmp_path / "data" / "raw" / "video_licensed"
    licensed_root.mkdir(parents=True)
    manifest = tmp_path / "manifest.csv"
    approved_path = licensed_root / "world_cup" / "clip.mp4"
    approved_path.parent.mkdir(parents=True)
    approved_path.touch()

    manifest.write_text(
        "video_id,match_id,clip_type,local_path,rights_status,rights_reference\n"
        f"ok,match_1,full_match,{approved_path},licensed,license-file:///perms/wc2026.pdf\n"
        f"skip,match_2,goal_montage,{approved_path},pending,\n",
        encoding="utf-8",
    )

    plan = build_processing_plan(manifest, licensed_root)
    assert [row.video_id for row in plan.processable] == ["ok"]
    assert [row.video_id for row in plan.skipped] == ["skip"]


def test_assert_processable_requires_rights_and_matching_path(tmp_path: Path) -> None:
    licensed_root = tmp_path / "licensed"
    licensed_root.mkdir()
    clip = licensed_root / "clip.mp4"
    clip.touch()
    other = tmp_path / "other.mp4"
    other.touch()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "video_id,match_id,clip_type,local_path,rights_status,rights_reference\n"
        f"ok,match_1,full_match,{clip},licensed,license-file:///perms/wc2026.pdf\n"
        f"unproven,match_2,full_match,{clip},licensed,\n",
        encoding="utf-8",
    )

    row = assert_processable(manifest, "ok", clip, licensed_root)
    assert row.video_id == "ok"

    with pytest.raises(ValueError):
        assert_processable(manifest, "unproven", clip, licensed_root)
    with pytest.raises(ValueError):
        assert_processable(manifest, "missing", clip, licensed_root)
    with pytest.raises(ValueError):
        assert_processable(manifest, "ok", other, licensed_root)


def test_validate_rejects_youtube_source(tmp_path: Path) -> None:
    licensed_root = tmp_path / "licensed"
    licensed_root.mkdir()
    clip = licensed_root / "clip.mp4"
    clip.touch()

    row = _row(clip, source_url="https://www.youtube.com/watch?v=abc123")
    with pytest.raises(ValueError, match="blocked modality"):
        validate_processable_video(row, licensed_root)


def test_validate_passes_licensed_local_file(tmp_path: Path) -> None:
    licensed_root = tmp_path / "licensed"
    licensed_root.mkdir()
    clip = licensed_root / "clip.mp4"
    clip.touch()

    validate_processable_video(_row(clip), licensed_root)


def test_validate_rejects_http_and_stream_sources(tmp_path: Path) -> None:
    licensed_root = tmp_path / "licensed"
    licensed_root.mkdir()
    clip = licensed_root / "clip.mp4"
    clip.touch()

    with pytest.raises(ValueError, match="blocked modality"):
        validate_processable_video(_row(clip, source_url="http://example.com/source"), licensed_root)
    with pytest.raises(ValueError, match="blocked modality"):
        validate_processable_video(_row(clip, source_url="rtmp://cdn.example.com/live/stream"), licensed_root)


def test_assert_processable_rejects_remote_modality(tmp_path: Path) -> None:
    licensed_root = tmp_path / "licensed"
    licensed_root.mkdir()
    clip = licensed_root / "clip.mp4"
    clip.touch()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "video_id,match_id,clip_type,source_url,local_path,rights_status,rights_reference\n"
        f"remote,match_1,full_match,https://youtu.be/abc,{clip},licensed,license-file:///perms/wc2026.pdf\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="blocked modality"):
        assert_processable(manifest, "remote", clip, licensed_root)
