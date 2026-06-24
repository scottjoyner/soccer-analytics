import pytest

from soccer_edge.models.game_state import GameStateFeatureSpec, GameStateTrainingConfig


def test_game_state_feature_spec_shape() -> None:
    spec = GameStateFeatureSpec(
        sequence_length=30,
        spatial_channels=6,
        pitch_height_bins=32,
        pitch_width_bins=48,
        tabular_features=12,
        output_classes=3,
    )
    spec.validate()
    assert spec.spatial_tensor_shape == (30, 6, 32, 48)


def test_game_state_feature_spec_rejects_invalid_sequence() -> None:
    spec = GameStateFeatureSpec(
        sequence_length=0,
        spatial_channels=6,
        pitch_height_bins=32,
        pitch_width_bins=48,
        tabular_features=12,
    )
    with pytest.raises(ValueError):
        spec.validate()


def test_training_config_defaults() -> None:
    config = GameStateTrainingConfig()
    config.validate()
    assert config.batch_size == 32
