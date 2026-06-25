import numpy as np
import pytest

from soccer_edge.models.dataset import RollingGameStateDataset, RollingGameStateSample
from soccer_edge.models.torch_optional import MissingTorchError, torch


def test_rolling_game_state_sample_shapes() -> None:
    sample = RollingGameStateSample(
        spatial_sequence=np.zeros((5, 3, 8, 8), dtype=np.float32),
        tabular_sequence=np.zeros((5, 4), dtype=np.float32),
        label=1,
    )
    assert sample.spatial_sequence.shape == (5, 3, 8, 8)
    assert sample.tabular_sequence.shape == (5, 4)


def test_rolling_dataset_optional_torch() -> None:
    sample = RollingGameStateSample(
        spatial_sequence=np.zeros((5, 3, 8, 8), dtype=np.float32),
        tabular_sequence=np.zeros((5, 4), dtype=np.float32),
        label=1,
    )
    if torch is None:
        with pytest.raises(MissingTorchError):
            RollingGameStateDataset([sample])
        return

    dataset = RollingGameStateDataset([sample])
    item = dataset[0]
    assert tuple(item["spatial_sequence"].shape) == (5, 3, 8, 8)
    assert int(item["label"]) == 1
