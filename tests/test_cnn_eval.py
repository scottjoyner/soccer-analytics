"""Tests for the out-of-sample CNN highlight-clip evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soccer_edge.evaluation import cnn_eval
from soccer_edge.evaluation.cnn_eval import (
    evaluate_cnn_out_of_sample,
    evaluate_cnn_repeated_cv,
)
from soccer_edge.models.torch_optional import torch


class _TrivialModel(torch.nn.Module):
    def __init__(self, output_classes: int = 3):
        super().__init__()
        self.fc = torch.nn.Linear(1, output_classes)

    def forward(self, x):
        return self.fc(x.new_zeros(x.shape[0], 1))


def _patch_training(monkeypatch) -> None:
    def fake_train(npz_path, output_dir, output_classes=3, epochs=1, batch_size=4, **kwargs):
        return {"model": str(output_dir / "fake_model.pt")}

    def fake_load(bundle_dir):
        return _TrivialModel(), {}

    monkeypatch.setattr(cnn_eval, "train_cnn_from_npz", fake_train)
    monkeypatch.setattr(cnn_eval, "load_bundle", fake_load)


def _make_results(n_matches: int) -> pd.DataFrame:
    rows = []
    for i in range(n_matches):
        home, away = (2, 1) if i % 3 else (1, 1)
        winner = "home" if home > away else ("away" if away > home else "draw")
        rows.append(
            {"match_id": f"M{i:03d}", "home_score": home, "away_score": away, "winner": winner}
        )
    return pd.DataFrame(rows)


def _make_detections(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = 12
    rows = []
    for f in range(frames):
        for _ in range(rng.integers(2, 6)):
            x1, y1 = rng.integers(0, 1600), rng.integers(0, 900)
            rows.append(
                {
                    "frame_idx": f,
                    "class_name": "player",
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(min(x1 + 40, 1920)),
                    "y2": float(min(y1 + 40, 1080)),
                }
            )
        rows.append(
            {
                "frame_idx": f,
                "class_name": "ball",
                "x1": 100.0,
                "y1": 100.0,
                "x2": 120.0,
                "y2": 120.0,
            }
        )
    return pd.DataFrame(rows)


def test_evaluate_cnn_out_of_sample_smoke(tmp_path) -> None:
    if torch is None:
        pytest.skip("torch not installed")
    results = _make_results(9)
    detections = {f"M{i:03d}": _make_detections(i) for i in range(9)}
    metrics = evaluate_cnn_out_of_sample(
        results, detections, tmp_path, epochs=1, batch_size=4
    )
    assert metrics["n_train_matches"] + metrics["n_test_matches"] == 9
    assert metrics["n_train_sequences"] > 0
    assert metrics["n_test_sequences"] > 0
    assert 0.0 <= metrics["sequence_accuracy"] <= 1.0
    assert 0.0 <= metrics["match_accuracy"] <= 1.0
    assert metrics["winner_brier"] >= 0.0
    assert (tmp_path / "train_grid_samples.npz").exists()


def test_evaluate_cnn_repeated_cv_smoke(tmp_path, monkeypatch) -> None:
    if torch is None:
        pytest.skip("torch not installed")
    _patch_training(monkeypatch)
    results = _make_results(12)
    detections = {f"M{i:03d}": _make_detections(i) for i in range(12)}
    metrics = evaluate_cnn_repeated_cv(
        results, detections, tmp_path, n_splits=3, repeats=2, epochs=1, batch_size=4
    )
    assert metrics["n_splits"] == 3
    assert metrics["repeats"] == 2
    assert metrics["n_folds"] == 6
    assert len(metrics["per_fold"]) == 6
    for key in (
        "sequence_accuracy_mean",
        "sequence_accuracy_std",
        "match_accuracy_mean",
        "match_accuracy_std",
        "winner_brier_mean",
        "winner_brier_std",
        "sequence_baseline_accuracy_mean",
        "match_baseline_accuracy_mean",
    ):
        assert key in metrics
        assert isinstance(metrics[key], float)
    assert metrics["sequence_accuracy_std"] >= 0.0
    assert metrics["winner_brier_std"] >= 0.0
    for fold in metrics["per_fold"]:
        assert fold["n_train_matches"] + fold["n_test_matches"] == 12


def test_evaluate_cnn_repeated_cv_single_fold_fields(tmp_path, monkeypatch) -> None:
    if torch is None:
        pytest.skip("torch not installed")
    _patch_training(monkeypatch)
    results = _make_results(9)
    detections = {f"M{i:03d}": _make_detections(i) for i in range(9)}
    metrics = evaluate_cnn_repeated_cv(
        results, detections, tmp_path, n_splits=3, repeats=1, epochs=1, batch_size=4
    )
    assert metrics["n_folds"] == 3
    assert 0.0 <= metrics["sequence_accuracy_mean"] <= 1.0
    assert 0.0 <= metrics["match_accuracy_mean"] <= 1.0
    assert metrics["winner_brier_mean"] >= 0.0
