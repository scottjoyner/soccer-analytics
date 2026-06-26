from soccer_edge.media_inference import box_from_mapping, make_media_callback
from soccer_edge.media_samples import MediaSample


def test_box_from_mapping() -> None:
    sample = MediaSample(index=3, time_seconds=1.5, data=None)
    box = box_from_mapping(sample, {"label": "player", "score": 0.9, "x1": 1, "y1": 2, "x2": 3, "y2": 4})
    assert box.frame_idx == 3
    assert box.class_name == "player"
    assert box.confidence == 0.9


def test_make_media_callback() -> None:
    callback = make_media_callback(lambda _data: [{"class_name": "ball", "confidence": 0.8}])
    rows = callback(MediaSample(index=1, time_seconds=0.5, data=None))
    assert len(rows) == 1
    assert rows[0].class_name == "ball"
