import numpy as np
import pytest

from soccer_edge.models.rolling_dataset import RollingGameStateDataset, build_rolling_samples
from soccer_edge.models.torch_optional import MissingTorchError, torch


def test_build_rolling_samples() -> None:
    spatial = [np.zeros((3, 4, 5), dtype=np.float32) + idx for idx in range(4)]
    tabular = [np.array([idx], dtype=np.float32) for idx in range(4)]
    labels = [0, 1, 0, 1]

    samples = build_rolling_samples(spatial, tabular, labels, sequence_length=3)
    assert len(samples) == 2
    assert samples[0].spatial_window.shape == (3, 3, 4, 5)
    assert samples[0].label == 0
    assert samples[1].label == 1


def test_dataset_requires_torch_when_missing() -> None:
    samples = build_rolling_samples(
        [np.zeros((3, 4, 5), dtype=np.float32)],
        [np.zeros((2,), dtype=np.float32)],
        [1],
        sequence_length=1,
    )
    if torch is None:
        with pytest.raises(MissingTorchError):
            RollingGameStateDataset(samples)
    else:
        dataset = RollingGameStateDataset(samples)
        assert len(dataset) == 1
