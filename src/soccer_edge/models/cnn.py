from soccer_edge.models.torch_optional import nn, require_torch, torch


if nn is not None:

    class FieldStateCNN(nn.Module):
        def __init__(self, in_channels: int, output_classes: int = 3, hidden_size: int = 128) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_classes),
            )

        def forward(self, spatial_state: torch.Tensor) -> torch.Tensor:
            return self.head(self.encoder(spatial_state))

else:

    class FieldStateCNN:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            require_torch()
