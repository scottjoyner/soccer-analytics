from pathlib import Path

import numpy as np

from soccer_edge.models.bundle import save_bundle
from soccer_edge.models.cnn import FieldStateCNN
from soccer_edge.models.cnn_training import train_field_state_cnn
from soccer_edge.models.dataset import RollingGameStateDataset, RollingGameStateSample
from soccer_edge.models.game_state import GameStateTrainingConfig
from soccer_edge.models.torch_optional import require_torch, torch


def load_tensor_samples(npz_path: Path) -> list[RollingGameStateSample]:
    data = np.load(npz_path)
    spatial = data["spatial"]
    labels = data["labels"]
    tabular = data["tabular"] if "tabular" in data.files else np.zeros((len(labels), spatial.shape[1], 0), dtype=np.float32)
    samples: list[RollingGameStateSample] = []
    for idx in range(len(labels)):
        samples.append(
            RollingGameStateSample(
                spatial_sequence=spatial[idx].astype(np.float32),
                tabular_sequence=tabular[idx].astype(np.float32),
                label=int(labels[idx]),
            )
        )
    return samples


def train_cnn_from_npz(
    npz_path: Path,
    output_dir: Path,
    output_classes: int = 3,
    epochs: int = 1,
    batch_size: int = 4,
    hidden_size: int = 128,
) -> dict[str, Path]:
    require_torch()
    samples = load_tensor_samples(npz_path)
    if not samples:
        raise ValueError("no samples found")
    channels = samples[0].spatial_sequence.shape[-3]
    dataset = RollingGameStateDataset(samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = FieldStateCNN(in_channels=channels, output_classes=output_classes, hidden_size=hidden_size)
    config = GameStateTrainingConfig(epochs=epochs, batch_size=batch_size, hidden_size=hidden_size)
    history = train_field_state_cnn(model, dataloader, config)
    metrics = {"final_loss": float(history.losses[-1]) if history.losses else 0.0, "batches_seen": float(history.batches_seen)}
    return save_bundle(model, output_dir, "field_state_cnn", "v0", ["spatial_grid"], metrics)
