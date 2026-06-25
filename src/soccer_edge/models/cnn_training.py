from dataclasses import dataclass

from soccer_edge.models.game_state import GameStateTrainingConfig
from soccer_edge.models.torch_optional import nn, require_torch, torch


@dataclass(frozen=True)
class TrainingHistory:
    losses: list[float]
    batches_seen: int = 0


def unpack_cnn_batch(batch):
    if isinstance(batch, dict):
        return batch["spatial_sequence"], batch["label"]
    spatial, _tabular, labels = batch
    return spatial, labels


def train_field_state_cnn(model, dataloader, config: GameStateTrainingConfig, device: str = "cpu") -> TrainingHistory:
    require_torch()
    config.validate()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    batches_seen = 0

    for _ in range(config.epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in dataloader:
            spatial, labels = unpack_cnn_batch(batch)
            spatial = spatial.to(device)
            labels = labels.to(device)
            if spatial.ndim == 5:
                spatial = spatial[:, -1, :, :, :]
            optimizer.zero_grad()
            logits = model(spatial)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batch_count += 1
            batches_seen += 1
        if batch_count:
            losses.append(epoch_loss / batch_count)
    return TrainingHistory(losses=losses, batches_seen=batches_seen)
