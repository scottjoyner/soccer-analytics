import numpy as np
import pandas as pd

from soccer_edge.evaluation.match_eval import repeated_cv_match_predictor


def _dummy_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "home_xt": rng.random(n),
            "away_xt": rng.random(n),
            "home_xg": rng.random(n),
            "away_xg": rng.random(n),
            "winner": rng.integers(0, 3, n),
            "home_score": rng.integers(0, 5, n),
            "away_score": rng.integers(0, 5, n),
        }
    )


def test_repeated_cv_returns_metrics() -> None:
    frame = _dummy_frame()
    metrics = repeated_cv_match_predictor(frame, ["home_xt", "away_xt"])
    for key in [
        "n_matches",
        "winner_accuracy_mean",
        "winner_accuracy_std",
        "winner_brier_mean",
        "home_score_mse_mean",
        "away_score_mse_mean",
        "majority_baseline_accuracy_mean",
    ]:
        assert key in metrics
    assert metrics["n_matches"] == 60


def test_repeated_cv_feature_refit_fn() -> None:
    frame = _dummy_frame()
    calls = []

    def _refit(train_idx, test_idx, frm):
        calls.append((len(train_idx), len(test_idx)))
        return frm

    metrics = repeated_cv_match_predictor(
        frame, ["home_xt", "away_xt"], feature_refit_fn=_refit
    )
    assert len(calls) == 5 * 10  # n_splits * n_repeats
    assert metrics["winner_accuracy_mean"] >= 0.0
