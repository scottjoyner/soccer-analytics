"""Realtime win-probability, match-state, and actionable triggers for live footage.

This subpackage turns a stream of YOLO detections (from ``capture --detect`` or the
``live watch`` command) into a *live* picture of a match:

* ``realtime_state``   - frame-by-frame state management: rolling frame buffer,
  fps throttling, track/ball continuity across frames, and a low-confidence review
  queue (the same data the offline pipeline would route to human review).
* ``live_state``      - aggregates the rolling window into a per-window match state
  (possession share, territory, pressure rate, an xT proxy) without needing pitch
  calibration (players are treated as undifferentiated when uncalibrated).
* ``win_prob``        - updates an expected-winner probability every window from the
  CV features, either by scoring the live feature vector against a trained
  match-outcome bundle, or by a calibration-free Elo-like drift when no bundle is
  available.
* ``triggers``       - emits *actionable* alerts: "expected winner" once the
  probability clears a confidence/gap threshold, and "momentum"/"comeback" alerts
  when the win-prob moves sharply or a side climbs back from a deficit.

Everything is rights-gated: callers must pass an approved ``video_id``/manifest row,
and public URLs are never accepted as inputs (enforced upstream by
``assert_local_input``).
"""

from soccer_edge.realtime.live_state import (
    LiveMatchState,
    aggregate_window_state,
)
from soccer_edge.realtime.realtime_state import (
    RealtimeDetector,
    RealtimeState,
    ReviewQueueEntry,
)
from soccer_edge.realtime.triggers import (
    Trigger,
    TriggerConfig,
    TriggerEngine,
    evaluate_triggers,
)
from soccer_edge.realtime.win_prob import (
    LiveWinProbability,
    WinProbConfig,
    elo_drift_win_prob,
)

__all__ = [
    "RealtimeState",
    "RealtimeDetector",
    "ReviewQueueEntry",
    "LiveMatchState",
    "aggregate_window_state",
    "LiveWinProbability",
    "WinProbConfig",
    "elo_drift_win_prob",
    "Trigger",
    "TriggerConfig",
    "TriggerEngine",
    "evaluate_triggers",
]
