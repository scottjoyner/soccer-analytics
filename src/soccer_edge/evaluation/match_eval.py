"""Repeated cross-validated evaluation for match-outcome models.

Used to report mean +/- std out-of-sample metrics (accuracy, Brier, score MSE)
with a proper majority-class baseline, so a null result is statistically
defensible rather than an artifact of one lucky split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def repeated_cv_match_predictor(
    frame: pd.DataFrame,
    feature_columns: list[str],
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 0,
) -> dict:
    """Repeated stratified CV of a calibrated winner classifier + score regressors.

    Returns mean and std of test accuracy, winner Brier, and home/away score MSE,
    plus the mean majority-class baseline accuracy.
    """
    x = frame[feature_columns].to_numpy(dtype=float)
    y = frame["winner"].to_numpy(dtype=int)
    home = frame["home_score"].to_numpy(dtype=float)
    away = frame["away_score"].to_numpy(dtype=float)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    acc, brier, home_mse, away_mse, base_acc = [], [], [], [], []

    for train_idx, test_idx in cv.split(x, y):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LogisticRegression(max_iter=1000), cv=3
                    ),
                ),
            ]
        )
        pipe.fit(x_train, y_train)
        proba = pipe.predict_proba(x_test)
        classes = pipe.classes_
        y_test_arr = np.asarray(y_test)
        mapped = np.zeros((len(y_test_arr), len(classes)))
        for j, c in enumerate(classes):
            mapped[:, j] = (y_test_arr == c).astype(float)
        brier.append(brier_score_loss(mapped.ravel(), proba.ravel()))
        acc.append(float((pipe.predict(x_test) == y_test).mean()))

        majority = pd.Series(y_train).mode().iloc[0]
        base_acc.append(float((np.full(len(y_test), majority) == y_test).mean()))

        for target, store in ((home, home_mse), (away, away_mse)):
            reg = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
            reg.fit(x_train, target[train_idx])
            pred = reg.predict(x_test)
            store.append(float(np.mean((pred - target[test_idx]) ** 2)))

    def _mean_std(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values)
        return float(arr.mean()), float(arr.std())

    acc_m, acc_s = _mean_std(acc)
    brier_m, brier_s = _mean_std(brier)
    home_m, home_s = _mean_std(home_mse)
    away_m, away_s = _mean_std(away_mse)
    base_m, base_s = _mean_std(base_acc)

    return {
        "n_matches": int(len(frame)),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "winner_accuracy_mean": acc_m,
        "winner_accuracy_std": acc_s,
        "winner_brier_mean": brier_m,
        "winner_brier_std": brier_s,
        "majority_baseline_accuracy_mean": base_m,
        "majority_baseline_accuracy_std": base_s,
        "home_score_mse_mean": home_m,
        "home_score_mse_std": home_s,
        "away_score_mse_mean": away_m,
        "away_score_mse_std": away_s,
    }
