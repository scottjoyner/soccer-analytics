from soccer_edge.models.torch_optional import nn, require_torch, torch


if nn is not None:

    class HybridCNNTemporalModel(nn.Module):
        def __init__(self, spatial_channels: int, tabular_features: int, output_classes: int = 3, hidden_size: int = 128) -> None:
            super().__init__()
            self.frame_encoder = nn.Sequential(
                nn.Conv2d(spatial_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.sequence_head = nn.Sequential(
                nn.Linear(32 + tabular_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_classes),
            )

        def forward(self, spatial_sequence: torch.Tensor, tabular_sequence: torch.Tensor) -> torch.Tensor:
            batch_size, sequence_length, channels, height, width = spatial_sequence.shape
            flat = spatial_sequence.reshape(batch_size * sequence_length, channels, height, width)
            encoded = self.frame_encoder(flat).reshape(batch_size, sequence_length, -1)
            combined = torch.cat([encoded, tabular_sequence], dim=-1)
            pooled = combined.mean(dim=1)
            return self.sequence_head(pooled)

else:

    class HybridCNNTemporalModel:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()
