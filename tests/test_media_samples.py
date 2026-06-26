import pytest

from soccer_edge.media_reader import MissingMediaReaderError
from soccer_edge.media_samples import iter_media_samples


def test_iter_media_samples_missing_or_bad_file(tmp_path) -> None:
    bad_path = tmp_path / "missing.mp4"
    try:
        samples = list(iter_media_samples(bad_path, max_samples=1))
    except MissingMediaReaderError:
        pytest.skip("optional reader dependency is not installed")
    except ValueError:
        return
    assert samples == []


def test_iter_media_samples_requires_positive_stride(tmp_path) -> None:
    with pytest.raises(ValueError):
        list(iter_media_samples(tmp_path / "missing.mp4", stride=0))
