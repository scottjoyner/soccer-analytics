from soccer_edge.video.local_pipeline import run_local_video_pipeline


def test_run_local_video_pipeline(tmp_path) -> None:
    input_path = tmp_path / "clip.mp4"
    input_path.write_bytes(b"demo")
    result = run_local_video_pipeline(input_path, tmp_path / "out", frame_count=2)
    assert result.frame_count == 2
    assert result.table_paths["detections"].exists()
