"""Live win-probability that updates every rolling window from CV features.

Two operating modes, both returning a 3-vector ``[p_home, p_draw, p_away]``:

* **logistic bundle** (``method="logistic_bundle"``) - when a trained
  match-outcome bundle exists, the live feature vector is scored with
  ``model.predict_proba``. This is the *same* model the offline
  ``train match-predictor`` produces, so a live clip and the batch pipeline share
  one notion of "who is winning".
* **elo drift** (``method="elo"`` or when no bundle is configured) - a
  calibration-free, interpretable fallback: a prior (e.g. home 0.5) is nudged
  each window by the live signals (territory, possession, pressure, xT proxy).
  It never needs a trained artifact, which matters for a rights-clean local setup,
  and it degrades gracefully when only CV features are available.

Both modes apply exponential smoothing across windows so a single noisy frame does not
whiplash the probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from soccer_edge.models.bundle import load_bundle
from soccer_edge.realtime.live_state import LiveMatchState


@dataclass
class WinProbConfig:
    method: str = "elo"  # "elo" | "logistic_bundle"
    bundle_dir: Path | None = None
    prior_home: float = 0.5
    prior_draw: float = 0.2
    prior_away: float = 0.3
    smoothing: float = 0.25  # 0 = no smoothing, 1 = frozen
    min_observations: int = 5
    feature_names: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> WinProbConfig:
        bundle = data.get("bundle_dir")
        return cls(
            method=data.get("method", "elo"),
            bundle_dir=Path(bundle) if bundle else None,
            prior_home=float(data.get("prior_home", 0.5)),
            prior_draw=float(data.get("prior_draw", 0.2)),
            prior_away=float(data.get("prior_away", 0.3)),
            smoothing=float(data.get("smoothing", 0.25)),
            min_observations=int(data.get("min_observations", 5)),
            feature_names=list(data["feature_names"]) if data.get("feature_names") else None,
        )


def elo_drift_win_prob(
    state: LiveMatchState,
    *,
    prior_home: float = 0.5,
    prior_draw: float = 0.2,
    prior_away: float = 0.3,
) -> np.ndarray:
    """Calibration-free expected-winner from live geometry (3-vector)."""

    home_ev = 0.45 * state.territory + 0.35 * state.possession_share + 0.20 * state.pressure_rate + 0.10 * state.xt_proxy
    away_ev = (
        0.45 * (1.0 - state.territory)
        + 0.35 * (1.0 - state.possession_share)
        + 0.10 * (1.0 - state.pressure_rate)
        + 0.10 * (1.0 - state.xt_proxy)
    )
    home_ev = float(min(1.0, max(0.0, home_ev)))
    away_ev = float(min(1.0, max(0.0, away_ev)))

    raw_home = prior_home + 0.5 * (home_ev - away_ev)
    raw_away = prior_away + 0.5 * (away_ev - home_ev)
    raw_draw = prior_draw
    total = raw_home + raw_away + raw_draw
    return np.array([raw_home / total, raw_draw / total, raw_away / total])


class LiveWinProbability:
    """Stateful live win-probability tracker."""

    def __init__(self, config: WinProbConfig | None = None) -> None:
        self.config = config or WinProbConfig()
        self._bundle = None
        self._bundle_meta = None
        if self.config.method == "logistic_bundle" and self.config.bundle_dir is not None:
            self._bundle, self._bundle_meta = load_bundle(Path(self.config.bundle_dir))
        self._smoothed: np.ndarray | None = None
        self.observations = 0

    def update(self, state: LiveMatchState) -> np.ndarray:
        if self.config.method == "logistic_bundle" and self._bundle is not None:
            proba = self._score_bundle(state)
        else:
            proba = elo_drift_win_prob(
                state,
                prior_home=self.config.prior_home,
                prior_draw=self.config.prior_draw,
                prior_away=self.config.prior_away,
            )
        self.observations += 1
        if self._smoothed is None or self.observations < self.config.min_observations:
            self._smoothed = proba.copy()
        else:
            a = self.config.smoothing
            self._smoothed = a * self._smoothed + (1.0 - a) * proba
        return self._smoothed.copy()

    def _score_bundle(self, state: LiveMatchState) -> np.ndarray:
        feature_names = self.config.feature_names or self._bundle_meta.feature_names
        row = state.to_feature_row()
        missing = [c for c in feature_names if c not in row]
        if missing:
            # Fall back to the drift model when the live features lack the bundle's columns.
            return elo_drift_win_prob(
                state,
                prior_home=self.config.prior_home,
                prior_draw=self.config.prior_draw,
                prior_away=self.config.prior_away,
            )
        x = pd.DataFrame([{c: float(row[c]) for c in feature_names}])
        if hasattr(self._bundle, "predict_proba"):
            return np.asarray(self._bundle.predict_proba(x))[0]
        pred = float(self._bundle.predict(x)[0])
        return np.array([1.0 - pred, 0.0, pred]) if pred < 0.5 else np.array([0.0, 0.0, 1.0])

    @property
    def current(self) -> np.ndarray:
        if self._smoothed is None:
            return np.array([self.config.prior_home, self.config.prior_draw, self.config.prior_away])
        return self._smoothed.copy()

    def expected_winner(self) -> str:
        idx = int(np.argmax(self.current))
        return ("home", "draw", "away")[idx]
