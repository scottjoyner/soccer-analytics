from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from soccer_edge.media_samples import MediaSample


@dataclass(frozen=True)
class MediaBox:
    frame_idx: int
    timestamp_seconds: float
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def box_from_mapping(sample: MediaSample, item: dict[str, Any]) -> MediaBox:
    return MediaBox(
        frame_idx=sample.index,
        timestamp_seconds=sample.time_seconds,
        class_name=str(item.get("class_name", item.get("label", "object"))),
        confidence=float(item.get("confidence", item.get("score", 0.0))),
        x1=float(item.get("x1", 0.0)),
        y1=float(item.get("y1", 0.0)),
        x2=float(item.get("x2", 0.0)),
        y2=float(item.get("y2", 0.0)),
    )


def make_media_callback(runner: Callable[[object], list[dict[str, Any]]]) -> Callable[[MediaSample], list[MediaBox]]:
    def callback(sample: MediaSample) -> list[MediaBox]:
        return [box_from_mapping(sample, item) for item in runner(sample.data)]

    return callback
