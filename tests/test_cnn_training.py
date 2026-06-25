import numpy as np
import pytest

from soccer_edge.models.cnn import FieldStateCNN
from soccer_edge.models.cnn_training import train_field_state_cnn, unpack_cnn_batch
from soccer_edge.models.dataset import RollingGameStateDataset, RollingGameStateSample
from soccer_edge.models.game_state import GameStateTrainingConfig
from soccer_edge.models.torch_optional import MissingTorchError, torch


def test_cnn_training_optional_torch() -> None:
    sample = RollingGameStateSample(
        spatial_sequence=np.zeros((2, 3, 8, 8), dtype=np.float32),
        tabular_sequence=np.zeros((2, 1), dtype=np.float32),
        label=1,
    )
    config = GameStateTrainingConfig(epochs=1, batch_size=1)

    if torch is None:
        with pytest.raises(MissingTorchError):
            RollingGameStateDataset([sample])
        return

    dataset = RollingGameStateDataset([sample])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = FieldStateCNN(in_channels=3, output_classes=3, hidden_size=16)
    history = train_field_state_cnn(model, dataloader, config)
    assert len(history.losses) == 1
    assert history.batches_seen == 1


def test_unpack_cnn_batch_from_tuple_when_torch_available() -> None:
    if torch is None:
        pytest.skip("torch is optional")
    batch = (torch.zeros((1, 1, 3, 4, 5)), torch.zeros((1, 1, 2)), torch.tensor([0]))
    spatial, labels = unpack_cnn_batch(batch)
    assert tuple(spatial.shape) == (1, 1, 3, 4, 5)
    assert labels.item() == 0
