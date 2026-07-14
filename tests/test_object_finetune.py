import importlib.util
from pathlib import Path

import pandas as pd
import pytest

if importlib.util.find_spec("cv2") is None or importlib.util.find_spec("ultralytics") is None:
    pytest.skip("opencv/ultralytics not installed", allow_module_level=True)


def _write_tiny_video(path: Path) -> None:
    import cv2 as _cv2
    import numpy as np

    w, h, n = 320, 240, 12
    writer = _cv2.VideoWriter(str(path), _cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(n):
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        _cv2.rectangle(frame, (40 + i * 10, 100), (80 + i * 10, 180), (200, 200, 200), -1)
        _cv2.circle(frame, (200, 60 + i * 5), 12, (0, 0, 255), -1)
        writer.write(frame)
    writer.release()


def test_player_ball_finetune_prep_chain(tmp_path) -> None:
    from soccer_edge.pipeline.object_finetune import run_player_ball_finetune

    video = tmp_path / "clip.mp4"
    _write_tiny_video(video)
    weights = Path("yolov8n.pt")
    if not weights.exists():
        pytest.skip("yolov8n.pt weights not present")

    outputs = run_player_ball_finetune(
        input_path=video,
        base_model_path=weights,
        output_dir=tmp_path / "finetune",
        max_frames=6,
        train_object_model=False,
    )
    assert outputs.frame_manifest.exists()
    assert outputs.detections.exists()
    assert outputs.annotation_config.exists()
    assert outputs.data_card.exists()
    annotations = pd.read_csv(outputs.train_annotations)
    assert "class_name" in annotations.columns
