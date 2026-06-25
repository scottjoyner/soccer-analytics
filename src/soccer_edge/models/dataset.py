from dataclasses import dataclass

import numpy as np

from soccer_edge.models.torch_optional import require_torch, torch


@dataclass(frozen=True)
class RollingGameStateSample:
    spatial_sequence: np.ndarray
    tabular_sequence: np.ndarray
    label: int


class RollingGameStateDataset:
    def __init__(self, samples: list[RollingGameStateSample]) -> None:
        require_torch()
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return {
            "spatial_sequence": torch.tensor(sample.spatial_sequence, dtype=torch.float32),
            "tabular_sequence": torch.tensor(sample.tabular_sequence, dtype=torch.float32),
            "label": torch.tensor(sample.label, dtype=torch.long),
        }
