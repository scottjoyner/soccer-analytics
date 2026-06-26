from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from soccer_edge.media_reader import require_media_reader


@dataclass(frozen=True)
class MediaSample:
    index: int
    time_seconds: float
    data: object


def iter_media_samples(input_path: Path, stride: int = 1, max_samples: int | None = None) -> Iterator[MediaSample]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    reader = require_media_reader()
    handle = reader.VideoCapture(str(input_path))
    if not handle.isOpened():
        raise ValueError(f"could not open input: {input_path}")
    rate = float(handle.get(reader.CAP_PROP_FPS) or 0.0)
    index = 0
    emitted = 0
    try:
        while True:
            ok, data = handle.read()
            if not ok:
                break
            if index % stride == 0:
                seconds = index / rate if rate > 0 else float(index)
                yield MediaSample(index=index, time_seconds=seconds, data=data)
                emitted += 1
                if max_samples is not None and emitted >= max_samples:
                    break
            index += 1
    finally:
        handle.release()
