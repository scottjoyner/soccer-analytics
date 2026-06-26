from pathlib import Path

import pandas as pd

from soccer_edge.models.bundle import load_bundle
from soccer_edge.models.cnn_runner import load_tensor_samples
from soccer_edge.models.dataset import RollingGameStateDataset
from soccer_edge.models.torch_optional import require_torch, torch


def export_cnn_bundle_predictions(
    bundle_dir: Path,
    npz_path: Path,
    output: Path,
    batch_size: int = 8,
    device: str = "cpu",
) -> Path:
    require_torch()
    model, _metadata = load_bundle(bundle_dir)
    samples = load_tensor_samples(npz_path)
    dataset = RollingGameStateDataset(samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model.to(device)
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in dataloader:
            spatial = batch["spatial_sequence"].to(device)
            labels = batch["label"].to(device)
            if spatial.ndim == 5:
                spatial = spatial[:, -1, :, :, :]
            logits = model(spatial)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            for row_idx in range(probabilities.shape[0]):
                row = {"label": int(labels[row_idx].cpu())}
                for class_idx in range(probabilities.shape[1]):
                    row[f"prob_{class_idx}"] = float(probabilities[row_idx, class_idx])
                rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
