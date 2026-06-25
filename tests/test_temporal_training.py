import numpy as np
import pytest

from soccer_edge.models.dataset import RollingGameStateDataset, RollingGameStateSample
from soccer_edge.models.game_state import GameStateTrainingConfig
from soccer_edge.models.hybrid import HybridCNNTemporalModel
from soccer_edge.models.temporal_training import train_temporal_model, unpack_temporal_batch
from soccer_edge.models.torch_optional import MissingTorchError, torch


def test_unpack_temporal_batch_from_dict_when_torch_available() -> None:
    if torch is None:
        pytest.skip("torch is optional")
    batch = {
        "spatial_sequence": torch.zeros((1, 2, 3, 8, 8)),
        "tabular_sequence": torch.zeros((1, 2, 4)),
        "label": torch.tensor([1]),
    }
    spatial, tabular, labels = unpack_temporal_batch(batch)
    assert tuple(spatial.shape) == (1, 2, 3, 8, 8)
    assert tuple(tabular.shape) == (1, 2, 4)
    assert labels.item() == 1


def test_temporal_training_optional_torch() -> None:
    sample = RollingGameStateSample(
        spatial_sequence=np.zeros((2, 3, 8, 8), dtype=np.float32),
        tabular_sequence=np.zeros((2, 4), dtype=np.float32),
        label=1,
    )
    config = GameStateTrainingConfig(epochs=1, batch_size=1)

    if torch is None:
        with pytest.raises(MissingTorchError):
            RollingGameStateDataset([sample])
        return

    dataset = RollingGameStateDataset([sample])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = HybridCNNTemporalModel(spatial_channels=3, tabular_features=4, output_classes=3, hidden_size=16)
    history = train_temporal_model(model, dataloader, config)
    assert len(history.losses) == 1
    assert history.batches_seen == 1
