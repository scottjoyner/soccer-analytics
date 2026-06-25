import pytest

from soccer_edge.models.hybrid import HybridCNNTemporalModel
from soccer_edge.models.torch_optional import MissingTorchError, torch


def test_hybrid_model_optional_torch() -> None:
    if torch is None:
        with pytest.raises(MissingTorchError):
            HybridCNNTemporalModel(spatial_channels=3, tabular_features=2)
        return

    model = HybridCNNTemporalModel(spatial_channels=3, tabular_features=2, output_classes=3, hidden_size=16)
    spatial = torch.zeros((2, 4, 3, 8, 8))
    tabular = torch.zeros((2, 4, 2))
    output = model(spatial, tabular)
    assert tuple(output.shape) == (2, 3)
