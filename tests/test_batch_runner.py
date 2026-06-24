from pathlib import Path

from soccer_edge.video.batch_runner import build_processing_plan


def test_build_processing_plan_filters_unapproved_rows(tmp_path: Path) -> None:
    licensed_root = tmp_path / "data" / "raw" / "video_licensed"
    licensed_root.mkdir(parents=True)
    manifest = tmp_path / "manifest.csv"
    approved_path = licensed_root / "world_cup" / "clip.mp4"
    approved_path.parent.mkdir(parents=True)
    approved_path.touch()

    manifest.write_text(
        "video_id,match_id,clip_type,local_path,rights_status\n"
        f"ok,match_1,full_match,{approved_path},licensed\n"
        f"skip,match_2,goal_montage,{approved_path},pending\n",
        encoding="utf-8",
    )

    plan = build_processing_plan(manifest, licensed_root)
    assert [row.video_id for row in plan.processable] == ["ok"]
    assert [row.video_id for row in plan.skipped] == ["skip"]
