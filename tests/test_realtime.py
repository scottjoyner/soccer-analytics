import numpy as np
import pandas as pd

from soccer_edge.realtime import (
    RealtimeDetector,
    RealtimeState,
    LiveMatchState,
    aggregate_window_state,
    LiveWinProbability,
    WinProbConfig,
    elo_drift_win_prob,
    TriggerEngine,
    TriggerConfig,
    evaluate_triggers,
)
from soccer_edge.video.detector import Detection


def _det(frame_idx, cls, x, conf=0.9):
    return Detection(frame_idx=frame_idx, class_name=cls, confidence=conf, x1=float(x), y1=500.0, x2=float(x) + 40, y2=540.0)


def test_realtime_state_matches_tracks_across_frames() -> None:
    st = RealtimeState()
    s1 = st.ingest(0, 0.0, [_det(0, "player", 100), _det(0, "ball", 120)])
    s2 = st.ingest(1, 0.5, [_det(1, "player", 115), _det(1, "ball", 135)])
    # Same player/ball should keep one track id across frames.
    assert s1[0].track_id == s2[0].track_id
    assert s1[1].track_id == s2[1].track_id
    assert len(st.ball_series) == 2
    assert st.review_queue == []  # conf 0.9 > review_threshold 0.4


def test_realtime_state_low_confidence_queue() -> None:
    st = RealtimeState(review_threshold=0.4, confidence_floor=0.25)
    st.ingest(0, 0.0, [_det(0, "player", 100, conf=0.35)])
    assert len(st.review_queue) == 1
    assert st.review_queue[0].confidence == 0.35


def test_realtime_state_throttle_skips_frames() -> None:
    st = RealtimeState(min_interval_seconds=1.0)
    st.ingest(0, 0.0, [_det(0, "player", 100)])
    skipped = st.ingest(1, 0.2, [_det(1, "player", 100)])
    assert skipped == []  # throttled
    kept = st.ingest(2, 1.5, [_det(2, "player", 100)])
    assert len(kept) == 1


def test_aggregate_window_state_basic() -> None:
    df = pd.DataFrame(
        [
            {"frame_idx": 0, "class_name": "player", "x1": 100, "y1": 500, "x2": 140, "y2": 540},
            {"frame_idx": 0, "class_name": "ball", "x1": 100, "y1": 500, "x2": 110, "y2": 510},
            {"frame_idx": 1, "class_name": "player", "x1": 900, "y1": 500, "x2": 940, "y2": 540},
            {"frame_idx": 1, "class_name": "ball", "x1": 900, "y1": 500, "x2": 910, "y2": 510},
        ]
    )
    live = aggregate_window_state(df, 0.0, 1.0, video_width=1920.0)
    assert live.n_player == 2
    assert live.n_ball == 2
    # Forward ball progress => positive xT proxy; territory in [0,1].
    assert live.xt_proxy > 0.0
    assert 0.0 <= live.territory <= 1.0
    assert 0.0 <= live.possession_share <= 1.0


def test_elo_drift_win_prob_sums_to_one() -> None:
    live = LiveMatchState(
        window_start=0, window_end=1, n_frames=1, n_player=1, n_ball=1,
        possession_share=0.8, territory=0.8, pressure_rate=0.5, xt_proxy=0.3,
        ball_center_x=0.8, ball_center_y=0.5,
    )
    p = elo_drift_win_prob(live, prior_home=0.5, prior_draw=0.2, prior_away=0.3)
    assert np.isclose(p.sum(), 1.0)
    assert p[0] > p[2]  # home dominates


def test_live_win_probability_smoothing_and_expected() -> None:
    wp = LiveWinProbability(WinProbConfig(method="elo", smoothing=0.5, min_observations=1))
    home_live = LiveMatchState(
        window_start=0, window_end=1, n_frames=1, n_player=1, n_ball=1,
        possession_share=0.8, territory=0.8, pressure_rate=0.4, xt_proxy=0.3,
        ball_center_x=0.8, ball_center_y=0.5,
    )
    p = wp.update(home_live)
    assert wp.expected_winner() == "home"
    assert p.shape == (3,)


def test_triggers_expected_winner_and_cooldown() -> None:
    engine = TriggerEngine(TriggerConfig(expected_winner_confidence=0.6, expected_winner_min_gap=0.1, cooldown_seconds=10.0))
    p_home = np.array([0.7, 0.15, 0.15])
    t1 = engine.observe(0.0, 1.0, p_home)
    assert len(t1) == 1
    assert t1[0].kind == "expected_winner"
    # Same side within cooldown -> no re-fire.
    t2 = engine.observe(2.0, 3.0, p_home)
    assert t2 == []
    # Side change re-fires.
    p_away = np.array([0.15, 0.15, 0.7])
    t3 = engine.observe(20.0, 21.0, p_away)
    assert any(t.kind == "expected_winner" and t.side == "away" for t in t3)


def test_triggers_momentum_and_comeback() -> None:
    cfg = TriggerConfig(momentum_window_seconds=100.0, momentum_min_shift=0.1, comeback_min_deficit=0.1)
    history = [
        (0.0, 1.0, np.array([0.2, 0.2, 0.6])),   # away dominant, home underdog
        (2.0, 3.0, np.array([0.2, 0.2, 0.6])),
        (20.0, 21.0, np.array([0.7, 0.15, 0.15])),  # home climbs back to favourite
    ]
    triggers = evaluate_triggers(history, cfg)
    kinds = {t.kind for t in triggers}
    assert "momentum" in kinds
    assert "comeback" in kinds
    assert any(t.side == "home" and t.kind == "comeback" for t in triggers)


def test_realtime_detector_calls_on_window(monkeypatch, tmp_path) -> None:
    class FakeDet:
        def __init__(self):
            self.i = 0

        def detect_frame(self, frame_idx, frame):
            self.i += 1
            x = 100.0 + (self.i * 50)
            return [
                _det(frame_idx, "player", x),
                _det(frame_idx, "ball", x),
            ]

    seen = []
    rt = RealtimeDetector(FakeDet(), window_seconds=1.0, on_window=lambda s, ws, we: seen.append((ws, we)))
    for f in range(6):
        rt.process_frame(None, timestamp_seconds=float(f) * 0.5)
    rt.flush()
    assert len(seen) >= 1  # at least one window closed
