from dataclasses import dataclass

import numpy as np

from soccer_edge.models.torch_optional import require_torch, torch


@dataclass(frozen=True)
class RollingSample:
    spatial_window: np.ndarray
    tabular_window: np.ndarray
    label: int


class RollingGameStateDataset:
    def __init__(self, samples: list[RollingSample]) -> None:
        require_torch()
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return (
            torch.tensor(sample.spatial_window, dtype=torch.float32),
            torch.tensor(sample.tabular_window, dtype=torch.float32),
            torch.tensor(sample.label, dtype=torch.long),
        )


def build_rolling_samples(
    spatial: list[np.ndarray],
    tabular: list[np.ndarray],
    labels: list[int],
    sequence_length: int,
) -> list[RollingSample]:
    if not (len(spatial) == len(tabular) == len(labels)):
        raise ValueError("spatial, tabular, and labels must have equal length")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    samples: list[RollingSample] = []
    for end_idx in range(sequence_length - 1, len(labels)):
        start_idx = end_idx - sequence_length + 1
        samples.append(
            RollingSample(
                spatial_window=np.stack(spatial[start_idx : end_idx + 1]),
                tabular_window=np.stack(tabular[start_idx : end_idx + 1]),
                label=labels[end_idx],
            )
        )
    return samples
