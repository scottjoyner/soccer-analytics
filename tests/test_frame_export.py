import pytest

from soccer_edge.frame_export import export_video_frame_manifest, frame_image_name
from soccer_edge.media_reader import MissingMediaReaderError


def test_frame_image_name() -> None:
    assert frame_image_name(12) == "frame_00000012.jpg"
    assert frame_image_name(1, prefix="clip", extension="png") == "clip_00000001.png"


def test_export_video_frame_manifest_missing_or_bad_file(tmp_path) -> None:
    try:
        path = export_video_frame_manifest(tmp_path / "missing.mp4", tmp_path / "frames", tmp_path / "frames.csv", max_frames=1)
    except MissingMediaReaderError:
        pytest.skip("optional reader dependency is not installed")
    except ValueError:
        return
    assert path.exists()
