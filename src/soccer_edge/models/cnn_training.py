from dataclasses import dataclass

from soccer_edge.models.game_state import GameStateTrainingConfig
from soccer_edge.models.torch_optional import nn, require_torch, torch


@dataclass(frozen=True)
class TrainingHistory:
    losses: list[float]


def train_field_state_cnn(model, dataloader, config: GameStateTrainingConfig, device: str = "cpu") -> TrainingHistory:
    require_torch()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []

    for _ in range(config.epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in dataloader:
            spatial = batch["spatial_sequence"].to(device)
            labels = batch["label"].to(device)
            if spatial.ndim == 5:
                spatial = spatial[:, -1, :, :, :]
            optimizer.zero_grad()
            logits = model(spatial)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batch_count += 1
        if batch_count:
            losses.append(epoch_loss / batch_count)
    return TrainingHistory(losses=losses)
