from dataclasses import dataclass


@dataclass(frozen=True)
class GameStateFeatureSpec:
    sequence_length: int
    spatial_channels: int
    pitch_height_bins: int
    pitch_width_bins: int
    tabular_features: int
    output_classes: int = 3

    def validate(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.spatial_channels <= 0:
            raise ValueError("spatial_channels must be positive")
        if self.pitch_height_bins <= 0 or self.pitch_width_bins <= 0:
            raise ValueError("pitch bins must be positive")
        if self.tabular_features < 0:
            raise ValueError("tabular_features cannot be negative")
        if self.output_classes <= 1:
            raise ValueError("output_classes must be greater than one")

    @property
    def spatial_tensor_shape(self) -> tuple[int, int, int, int]:
        return (
            self.sequence_length,
            self.spatial_channels,
            self.pitch_height_bins,
            self.pitch_width_bins,
        )


@dataclass(frozen=True)
class GameStateTrainingConfig:
    learning_rate: float = 0.0003
    batch_size: int = 32
    epochs: int = 20
    hidden_size: int = 128
    dropout: float = 0.2

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
