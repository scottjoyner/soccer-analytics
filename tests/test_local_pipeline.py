from soccer_edge.media_pipeline import run_media_table_stub
import pytest
from pathlib import Path

def test_run_local_video_pipeline_missing_file() -> None:
    from soccer_edge.video.local_pipeline import run_local_video_pipeline

    with pytest.raises(FileNotFoundError, match="not found"):
        run_local_video_pipeline(
            Path("/nonexistent/clip.mp4"),
            Path("out"),
            model_path="yolov8n.pt",
            enforce_rights=False,
        )




def test_run_local_pipeline_stub(tmp_path) -> None:
    input_path = tmp_path / "clip.mp4"
    input_path.write_bytes(b"demo")
    result = run_media_table_stub(input_path, tmp_path / "out", frame_count=2)
    assert result.frame_count == 2
    assert result.table_paths["detections"].exists()
