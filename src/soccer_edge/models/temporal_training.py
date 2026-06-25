from dataclasses import dataclass

from soccer_edge.models.game_state import GameStateTrainingConfig
from soccer_edge.models.torch_optional import nn, require_torch, torch


@dataclass(frozen=True)
class TemporalTrainingHistory:
    losses: list[float]
    batches_seen: int = 0


def unpack_temporal_batch(batch):
    if isinstance(batch, dict):
        return batch["spatial_sequence"], batch["tabular_sequence"], batch["label"]
    return batch


def train_temporal_model(model, dataloader, config: GameStateTrainingConfig, device: str = "cpu") -> TemporalTrainingHistory:
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
            spatial, tabular, labels = unpack_temporal_batch(batch)
            spatial = spatial.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(spatial, tabular)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batch_count += 1
            batches_seen += 1
        if batch_count:
            losses.append(epoch_loss / batch_count)
    return TemporalTrainingHistory(losses=losses, batches_seen=batches_seen)
