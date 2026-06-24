import pytest

from soccer_edge.models.cnn import FieldStateCNN
from soccer_edge.models.game_state import GameStateFeatureSpec
from soccer_edge.models.temporal_model import TemporalModelSpec
from soccer_edge.models.torch_optional import MissingTorchError, torch


def test_temporal_model_spec() -> None:
    spec = TemporalModelSpec(input_size=24, output_classes=3, hidden_size=64)
    spec.validate()
    assert spec.model_family == "recurrent"


def test_cnn_forward_shape_when_torch_available() -> None:
    if torch is None:
        with pytest.raises(MissingTorchError):
            FieldStateCNN(in_channels=3)
        return

    spec = GameStateFeatureSpec(
        sequence_length=1,
        spatial_channels=3,
        pitch_height_bins=16,
        pitch_width_bins=24,
        tabular_features=0,
        output_classes=3,
    )
    model = FieldStateCNN(in_channels=spec.spatial_channels, output_classes=spec.output_classes)
    x = torch.zeros((2, spec.spatial_channels, spec.pitch_height_bins, spec.pitch_width_bins))
    y = model(x)
    assert tuple(y.shape) == (2, 3)
