from pathlib import Path

from soccer_edge.local_finetune_pipeline import local_finetune_outputs


def test_local_finetune_outputs() -> None:
    outputs = local_finetune_outputs(Path("out"))
    assert outputs.frame_manifest == Path("out/frame_manifest.csv")
    assert outputs.detections == Path("out/video_model/detections.parquet")
    assert outputs.annotation_config == Path("out/annotations/yolo/data.yaml")
    assert outputs.data_card == Path("out/DATA_CARD.md")
