from soccer_edge.media_pipeline import run_media_table_stub


def test_run_local_pipeline_stub(tmp_path) -> None:
    input_path = tmp_path / "clip.mp4"
    input_path.write_bytes(b"demo")
    result = run_media_table_stub(input_path, tmp_path / "out", frame_count=2)
    assert result.frame_count == 2
    assert result.table_paths["detections"].exists()
