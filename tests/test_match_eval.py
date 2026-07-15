import numpy as np
import pandas as pd

from soccer_edge.evaluation.match_eval import repeated_cv_match_predictor


def _make_frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feats = {f"f{i}": rng.random(n) for i in range(6)}
    feats["winner"] = rng.integers(0, 3, n)
    feats["home_score"] = rng.integers(0, 5, n)
    feats["away_score"] = rng.integers(0, 5, n)
    return pd.DataFrame(feats)


def test_repeated_cv_returns_metrics() -> None:
    frame = _make_frame(120)
    metrics = repeated_cv_match_predictor(frame, [f"f{i}" for i in range(6)], n_splits=4, n_repeats=3)
    for key in [
        "winner_accuracy_mean",
        "winner_accuracy_std",
        "winner_brier_mean",
        "majority_baseline_accuracy_mean",
        "home_score_mse_mean",
        "away_score_mse_mean",
    ]:
        assert key in metrics
    assert 0.0 <= metrics["winner_brier_mean"] <= 1.0
    assert metrics["n_matches"] == 120
