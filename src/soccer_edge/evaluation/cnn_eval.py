"""Out-of-sample evaluation of the highlight-clip CNN winner classifier.

Splits matches into train/test (stratified by outcome), trains the existing
FieldStateCNN on the train grid sequences, and reports held-out accuracy at both
the sliding-window (sequence) level and the aggregated match level. This rounds
out the highlight-clip modeling story: the same null result the tabular features
show should also appear for the deep model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from soccer_edge.models.bundle import load_bundle
from soccer_edge.models.cnn_runner import train_cnn_from_npz
from soccer_edge.models.device import resolve_device
from soccer_edge.models.torch_optional import require_torch, torch
from soccer_edge.pipeline.match_predictor import (
    GRID_CHANNELS,
    GRID_HEIGHT,
    GRID_WIDTH,
    build_match_grid_table_multi,
    grid_column_names,
)

SEQUENCE_LENGTH = 4


def _build_sequences(
    grid_table: pd.DataFrame,
    spatial_columns: list[str],
    label_column: str,
    sequence_length: int = SEQUENCE_LENGTH,
    channels: int = GRID_CHANNELS,
    height: int = GRID_HEIGHT,
    width: int = GRID_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror tensor_samples.append_sequence_rows but also return match_ids."""
    samples: list[np.ndarray] = []
    labels: list[int] = []
    mids: list[str] = []
    for mid, group in grid_table.groupby("match_id", sort=False):
        values = group[spatial_columns].to_numpy(dtype=np.float32).reshape(len(group), channels, height, width)
        label_values = group[label_column].to_numpy(dtype=np.int64)
        for end_idx in range(sequence_length - 1, len(group)):
            start_idx = end_idx - sequence_length + 1
            samples.append(values[start_idx : end_idx + 1])
            labels.append(int(label_values[end_idx]))
            mids.append(mid)
    if not samples:
        raise ValueError("no CNN sequences produced (too few frames per match?)")
    return np.stack(samples).astype(np.float32), np.asarray(labels, dtype=np.int64), np.asarray(mids)


def _run_single_split(
    results: pd.DataFrame,
    detections_by_match: dict[str, pd.DataFrame],
    output_dir: Path,
    train_ids: list[str],
    test_ids: list[str],
    sequence_length: int = SEQUENCE_LENGTH,
    epochs: int = 5,
    batch_size: int = 8,
) -> dict:
    train_grid = build_match_grid_table_multi(
        results[results["match_id"].astype(str).isin(train_ids)],
        {m: detections_by_match[m] for m in train_ids},
    )
    test_grid = build_match_grid_table_multi(
        results[results["match_id"].astype(str).isin(test_ids)],
        {m: detections_by_match[m] for m in test_ids},
    )

    spatial = grid_column_names()
    train_seq, train_lab, _ = _build_sequences(train_grid, spatial, "winner", sequence_length)
    test_seq, test_lab, test_mids = _build_sequences(test_grid, spatial, "winner", sequence_length)

    train_npz = output_dir / "train_grid_samples.npz"
    np.savez(train_npz, spatial=train_seq.astype(np.float32), labels=train_lab.astype(np.int64))

    bundle = train_cnn_from_npz(
        train_npz,
        output_dir / "winner_cnn",
        output_classes=3,
        epochs=epochs,
        batch_size=batch_size,
    )
    model, _ = load_bundle(Path(bundle["model"]).parent)
    device = resolve_device()
    model.to(device)
    model.eval()

    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(test_seq), batch_size):
            x = torch.tensor(test_seq[i : i + batch_size, -1], dtype=torch.float32, device=device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(logits.argmax(1).detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

    seq_accuracy = float((preds == test_lab).mean())
    majority = int(pd.Series(train_lab).mode().iloc[0])
    base_seq_accuracy = float((np.full(len(test_lab), majority) == test_lab).mean())

    df = pd.DataFrame({"mid": test_mids, "pred": preds, "winner": test_lab})
    match_pred = df.groupby("mid")["pred"].agg(lambda s: int(s.mode().iloc[0]))
    match_true = df.groupby("mid")["winner"].first()
    match_accuracy = float((match_pred.to_numpy() == match_true.to_numpy()).mean())
    base_match_accuracy = float((np.full(len(match_true), majority) == match_true.to_numpy()).mean())

    n_classes = probs.shape[1]
    one_hot = np.zeros((len(test_lab), n_classes))
    one_hot[np.arange(len(test_lab)), test_lab] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return {
        "n_train_matches": len(train_ids),
        "n_test_matches": len(test_ids),
        "n_train_sequences": int(len(train_seq)),
        "n_test_sequences": int(len(test_seq)),
        "sequence_accuracy": seq_accuracy,
        "sequence_baseline_accuracy": base_seq_accuracy,
        "match_accuracy": match_accuracy,
        "match_baseline_accuracy": base_match_accuracy,
        "winner_brier": brier,
    }


def evaluate_cnn_out_of_sample(
    results: pd.DataFrame,
    detections_by_match: dict[str, pd.DataFrame],
    output_dir: Path,
    test_size: float = 0.3,
    sequence_length: int = SEQUENCE_LENGTH,
    epochs: int = 5,
    batch_size: int = 8,
    random_state: int = 0,
) -> dict:
    require_torch()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common = [m for m in results["match_id"].astype(str) if m in detections_by_match]
    sub = results[results["match_id"].astype(str).isin(common)].copy()
    sub["match_id"] = sub["match_id"].astype(str)
    strata = sub.set_index("match_id")["winner"].loc[common]
    train_ids, test_ids = train_test_split(
        common, test_size=test_size, stratify=strata, random_state=random_state
    )

    return _run_single_split(
        results,
        detections_by_match,
        output_dir,
        train_ids,
        test_ids,
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
    )


def evaluate_cnn_repeated_cv(
    results: pd.DataFrame,
    detections_by_match: dict[str, pd.DataFrame],
    output_dir: Path,
    n_splits: int = 5,
    repeats: int = 1,
    epochs: int = 5,
    batch_size: int = 8,
    random_state: int = 0,
) -> dict:
    require_torch()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common = [m for m in results["match_id"].astype(str) if m in detections_by_match]
    common = sorted(common)
    sub = results[results["match_id"].astype(str).isin(common)].copy()
    sub["match_id"] = sub["match_id"].astype(str)
    y = sub.set_index("match_id")["winner"].loc[common].to_numpy()
    x_idx = np.arange(len(common))

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=random_state)
    per_fold: list[dict] = []
    for k, (train_pos, test_pos) in enumerate(rskf.split(x_idx, y)):
        fold_dir = output_dir / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_ids = [common[i] for i in train_pos]
        test_ids = [common[i] for i in test_pos]
        fold_metrics = _run_single_split(
            results,
            detections_by_match,
            fold_dir,
            train_ids,
            test_ids,
            epochs=epochs,
            batch_size=batch_size,
        )
        per_fold.append(fold_metrics)

    def _mean_std(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    seq_acc_mean, seq_acc_std = _mean_std([f["sequence_accuracy"] for f in per_fold])
    seq_base_mean, seq_base_std = _mean_std([f["sequence_baseline_accuracy"] for f in per_fold])
    match_acc_mean, match_acc_std = _mean_std([f["match_accuracy"] for f in per_fold])
    match_base_mean, match_base_std = _mean_std([f["match_baseline_accuracy"] for f in per_fold])
    brier_mean, brier_std = _mean_std([f["winner_brier"] for f in per_fold])

    return {
        "n_splits": n_splits,
        "repeats": repeats,
        "n_folds": len(per_fold),
        "per_fold": per_fold,
        "sequence_accuracy_mean": seq_acc_mean,
        "sequence_accuracy_std": seq_acc_std,
        "sequence_baseline_accuracy_mean": seq_base_mean,
        "sequence_baseline_accuracy_std": seq_base_std,
        "match_accuracy_mean": match_acc_mean,
        "match_accuracy_std": match_acc_std,
        "match_baseline_accuracy_mean": match_base_mean,
        "match_baseline_accuracy_std": match_base_std,
        "winner_brier_mean": brier_mean,
        "winner_brier_std": brier_std,
    }
