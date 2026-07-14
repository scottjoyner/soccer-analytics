import pandas as pd
import pytest

from soccer_edge.video.local_catalog import discover_local_videos, write_local_video_catalog


def test_discover_local_videos(tmp_path) -> None:
    clip = tmp_path / "match one.mp4"
    clip.write_bytes(b"demo")
    rows = discover_local_videos(tmp_path, rights_status="owned", rights_reference="personal-recording://self")
    assert len(rows) == 1
    assert rows[0].video_id == "match_one"
    assert rows[0].rights_status == "owned"
    assert rows[0].rights_reference == "personal-recording://self"


def test_write_local_video_catalog(tmp_path) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"demo")
    output = write_local_video_catalog(
        tmp_path, tmp_path / "manifest.csv", rights_reference="personal-recording://self"
    )
    frame = pd.read_csv(output)
    assert len(frame) == 1
    assert frame.iloc[0]["rights_status"] == "owned"
    assert frame.iloc[0]["rights_reference"] == "personal-recording://self"


def test_discover_local_videos_rejects_unapproved_rights(tmp_path) -> None:
    with pytest.raises(ValueError):
        discover_local_videos(tmp_path, rights_status="pending")


def test_discover_local_videos_rejects_processable_without_rights(tmp_path) -> None:
    with pytest.raises(ValueError):
        discover_local_videos(tmp_path, rights_status="owned")
