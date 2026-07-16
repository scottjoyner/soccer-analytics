"""Actionable live triggers from the win-probability stream.

A small rules engine over the per-window win-probability + match state. It emits
**actionable** alerts the operator can act on while watching:

* ``expected_winner`` - fires once the smoothed probability for a side clears
  ``expected_winner_confidence`` AND the gap to the next-most-likely side exceeds
  ``expected_winner_min_gap``. (We do not spam this every window - a cooldown
  applies, and it only re-fires when the leader changes.)
* ``momentum`` - fires when the win-prob for the side in form moves by at least
  ``momentum_min_shift`` over ``momentum_window_seconds`` (computed from the
  rolling history of probabilities).
* ``comeback`` - fires when a side, previously the underdog by at least
  ``comeback_min_deficit``, climbs back to favourite.

All thresholds come from ``configs/live_triggers.json`` (the ``triggers`` block) and
are loaded by :class:`TriggerConfig`. History is kept in-memory so the engine can
compute momentum without external state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Trigger:
    kind: str  # "expected_winner" | "momentum" | "comeback"
    side: str  # "home" | "draw" | "away"
    window_start: float
    window_end: float
    probability: float
    message: str


@dataclass
class TriggerConfig:
    expected_winner_confidence: float = 0.6
    expected_winner_min_gap: float = 0.15
    momentum_window_seconds: float = 30.0
    momentum_min_shift: float = 0.12
    comeback_min_deficit: float = 0.2
    cooldown_seconds: float = 20.0

    @classmethod
    def from_dict(cls, data: dict) -> TriggerConfig:
        return cls(
            expected_winner_confidence=float(data.get("expected_winner_confidence", 0.6)),
            expected_winner_min_gap=float(data.get("expected_winner_min_gap", 0.15)),
            momentum_window_seconds=float(data.get("momentum_window_seconds", 30.0)),
            momentum_min_shift=float(data.get("momentum_min_shift", 0.12)),
            comeback_min_deficit=float(data.get("comeback_min_deficit", 0.2)),
            cooldown_seconds=float(data.get("cooldown_seconds", 20.0)),
        )


@dataclass
class TriggerEngine:
    config: TriggerConfig = field(default_factory=TriggerConfig)
    _history: list[tuple[float, float, np.ndarray]] = field(default_factory=list)
    _last_expected_at: float = -1.0
    _last_expected_side: str = ""

    def observe(self, window_start: float, window_end: float, prob: np.ndarray) -> list[Trigger]:
        prob = np.asarray(prob, dtype=float)
        self._history.append((window_start, window_end, prob))
        triggers: list[Trigger] = []
        triggers.extend(self._expected_winner(window_start, window_end, prob))
        triggers.extend(self._momentum(window_start, window_end, prob))
        triggers.extend(self._comeback(window_start, window_end, prob))
        return triggers

    def _expected_winner(self, start: float, end: float, prob: np.ndarray) -> list[Trigger]:
        order = np.argsort(prob)[::-1]
        leader = int(order[0])
        lead_p = float(prob[leader])
        runner_up = float(prob[order[1]])
        side = ("home", "draw", "away")[leader]
        if lead_p < self.config.expected_winner_confidence:
            return []
        if lead_p - runner_up < self.config.expected_winner_min_gap:
            return []
        # Cooldown + only re-fire on a side change.
        if end - self._last_expected_at < self.config.cooldown_seconds and side == self._last_expected_side:
            return []
        self._last_expected_at = end
        self._last_expected_side = side
        return [
            Trigger(
                kind="expected_winner",
                side=side,
                window_start=start,
                window_end=end,
                probability=lead_p,
                message=(
                    f"Expected winner: {side} (p={lead_p:.2f}, "
                    f"gap={lead_p - runner_up:.2f})"
                ),
            )
        ]

    def _momentum(self, start: float, end: float, prob: np.ndarray) -> list[Trigger]:
        if len(self._history) < 2:
            return []
        # Look back over the momentum window.
        window = [
            (s, e, p) for (s, e, p) in self._history if e >= start - self.config.momentum_window_seconds
        ]
        if len(window) < 2:
            return []
        first = window[0][2]
        last = window[-1][2]
        # The side that gained the most probability over the window is "in form".
        deltas = last - first
        side_idx = int(np.argmax(np.abs(deltas)))
        shift = float(deltas[side_idx])
        if abs(shift) < self.config.momentum_min_shift:
            return []
        side = ("home", "draw", "away")[side_idx]
        direction = "gaining" if shift > 0 else "losing"
        return [
            Trigger(
                kind="momentum",
                side=side,
                window_start=start,
                window_end=end,
                probability=float(last[side_idx]),
                message=f"Momentum: {side} {direction} p by {abs(shift):.2f} over the last {self.config.momentum_window_seconds:.0f}s",
            )
        ]

    def _comeback(self, start: float, end: float, prob: np.ndarray) -> list[Trigger]:
        if len(self._history) < 2:
            return []
        window = [
            (s, e, p) for (s, e, p) in self._history if e >= start - self.config.momentum_window_seconds
        ]
        if len(window) < 2:
            return []
        first = window[0][2]
        last = window[-1][2]
        # A comeback = a side that started as the underdog (lowest prob) and
        # ended as the favourite (highest prob), having gained >= deficit.
        start_underdog = int(np.argmin(first))
        end_favourite = int(np.argmax(last))
        gain = float(last[end_favourite] - first[end_favourite])
        if end_favourite == start_underdog and gain >= self.config.comeback_min_deficit:
            return [
                Trigger(
                    kind="comeback",
                    side=("home", "draw", "away")[end_favourite],
                    window_start=start,
                    window_end=end,
                    probability=float(last[end_favourite]),
                    message=(
                        f"Comeback: {('home', 'draw', 'away')[end_favourite]} "
                        f"climbed back from a {gain:.2f} deficit to favourite"
                    ),
                )
            ]
        return []


def evaluate_triggers(history: list[tuple[float, float, np.ndarray]], config: TriggerConfig | None = None) -> list[Trigger]:
    """Run an existing probability history through a fresh engine (for replay/tests)."""

    engine = TriggerEngine(config=config or TriggerConfig())
    out: list[Trigger] = []
    for start, end, prob in history:
        out.extend(engine.observe(start, end, prob))
    return out
