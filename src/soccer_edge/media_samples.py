from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

from soccer_edge.media_reader import require_media_reader

BLOCKED_SCHEMES = ("http", "https", "rtmp", "rtsp", "ftp")


def assert_local_input(input_path: Path) -> None:
    """Refuse to open remote/streaming URLs as detection inputs.

    Public video URLs are discovery metadata only (AGENTS.md hard safety rules)
    and must never be opened for inference. This guard runs even when the rights
    gate is bypassed for synthetic/pre-approved local frames, so a remote URL can
    never reach MediaSample iteration.
    """

    raw = str(input_path)
    parsed = urlparse(raw)
    if parsed.scheme and parsed.scheme.lower() in BLOCKED_SCHEMES:
        raise ValueError(
            f"refusing to open remote/streaming input {raw!r}: public URLs are "
            "discovery metadata only and must not be used as processing inputs."
        )
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")


@dataclass(frozen=True)
class MediaSample:
    index: int
    time_seconds: float
    data: object


def iter_media_samples(input_path: Path, stride: int = 1, max_samples: int | None = None) -> Iterator[MediaSample]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    assert_local_input(input_path)
    reader = require_media_reader()
    handle = reader.VideoCapture(str(input_path))
    if not handle.isOpened():
        raise ValueError(f"could not open input: {input_path}")
    rate = float(handle.get(reader.CAP_PROP_FPS) or 0.0)
    index = 0
    emitted = 0
    read_count = 0
    failed_after_frames = False
    try:
        while True:
            ok, data = handle.read()
            if not ok:
                if read_count > 0:
                    failed_after_frames = True
                break
            read_count += 1
            if index % stride == 0:
                seconds = index / rate if rate > 0 else float(index)
                yield MediaSample(index=index, time_seconds=seconds, data=data)
                emitted += 1
                if max_samples is not None and emitted >= max_samples:
                    break
            index += 1
    finally:
        handle.release()
    # A read that fails after frames were produced (rather than a clean EOF) often
    # means the stream was truncated/corrupt; surface it instead of silently
    # producing short, wrong detection tables.
    if failed_after_frames:
        warnings.warn(
            f"media stream ended early after {read_count} successful read(s); "
            "output may be truncated.",
            stacklevel=2,
        )
