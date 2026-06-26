import pytest

from soccer_edge.media_reader import MissingMediaReaderError, require_media_reader


def test_require_media_reader_returns_or_raises() -> None:
    try:
        module = require_media_reader()
    except MissingMediaReaderError:
        pytest.skip("optional reader dependency is not installed")
    assert module is not None
