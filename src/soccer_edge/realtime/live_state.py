"""Live match-state aggregation from a rolling window of detections.

Without pitch calibration and team assignment (a later enhancement), every detected
player is treated as undifferentiated. We still extract meaningful, rights-safe,
*relative* signals from the pixel geometry:

* **possession_share** - fraction of player detections belonging to the side that
  currently has more players in the attacking half (a proxy; replaced by true team
  assignment once calibration + team ID land).
* **territory** - mean normalized x of the ball (and players), so we know which
  third of the pitch the action is in.
* **pressure_rate** - fraction of frames in the window with a high defensive line
  (many players in the opponent half) - a crude pressure proxy.
* **xt_proxy** - the total "dangerous progress" of the ball: sum of forward
  (toward the opponent goal) normalized x-movement, which correlates with the xT
  surfaces built offline from StatsBomb events.

All features are normalized to ``[0, 1]`` so they feed both the offline
match-outcome model (via ``aggregate_window_state`` -> the same CV feature columns)
and the live win-probability drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

PLAYER_CLASSES = {"player", "person"}
BALL_CLASSES = {"ball", "sports ball"}


@dataclass
class LiveMatchState:
    window_start: float
    window_end: float
    n_frames: int
    n_player: int
    n_ball: int
    possession_share: float
    territory: float
    pressure_rate: float
    xt_proxy: float
    ball_center_x: float
    ball_center_y: float

    def to_feature_row(self) -> dict:
        """Map the live state onto the CV feature columns the match model expects."""

        return {
            "n_player": self.n_player,
            "n_ball": self.n_ball,
            "avg_det_per_frame": self.n_player / max(self.n_frames, 1),
            "ball_center_x": self.ball_center_x,
            "ball_center_y": self.ball_center_y,
            # Extra live-only signals (used by the drift model / triggers).
            "possession_share": self.possession_share,
            "territory": self.territory,
            "pressure_rate": self.pressure_rate,
            "xt_proxy": self.xt_proxy,
        }


def _classify(detections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    classes = detections["class_name"].astype(str).str.lower()
    players = detections[classes.isin(PLAYER_CLASSES)]
    balls = detections[classes.isin(BALL_CLASSES)]
    return players, balls


def aggregate_window_state(
    detections: pd.DataFrame,
    window_start: float,
    window_end: float,
    video_width: float = 1920.0,
    video_height: float = 1080.0,
) -> LiveMatchState:
    """Aggregate one rolling window of detection rows into a ``LiveMatchState``."""

    if detections is None or len(detections) == 0:
        return LiveMatchState(
            window_start=window_start,
            window_end=window_end,
            n_frames=0,
            n_player=0,
            n_ball=0,
            possession_share=0.5,
            territory=0.5,
            pressure_rate=0.0,
            xt_proxy=0.0,
            ball_center_x=0.0,
            ball_center_y=0.0,
        )

    players, balls = _classify(detections)
    n_frames = int(detections["frame_idx"].nunique()) if "frame_idx" in detections.columns else 1

    # Territory = mean normalized x of the ball (0 = own goal line, 1 = opp).
    if len(balls) > 0:
        ball_x = ((balls["x1"] + balls["x2"]) / 2.0).mean()
        ball_y = ((balls["y1"] + balls["y2"]) / 2.0).mean()
        territory = float(ball_x) / max(video_width, 1.0)
        ball_center_x = territory
        ball_center_y = float(ball_y) / max(video_height, 1.0)
    else:
        territory = 0.5
        ball_center_x = 0.0
        ball_center_y = 0.0

    # Possession proxy: which side has more players ahead of the halfway line.
    possession_share = 0.5
    if len(players) > 0:
        player_x = ((players["x1"] + players["x2"]) / 2.0) / max(video_width, 1.0)
        half = float(player_x.median())
        if half > 0:
            # >0.5 means the median player is in the attacking half.
            possession_share = float(min(0.95, 0.5 + (half - 0.5)))

    # Pressure proxy: share of frames where the ball is in the attacking third.
    pressure_rate = 0.0
    if len(balls) > 0 and "frame_idx" in balls.columns:
        attacking = balls.groupby("frame_idx").apply(
            lambda g: (((g["x1"] + g["x2"]) / 2.0 / max(video_width, 1.0)).mean() > 0.66).any()
        )
        pressure_rate = float(attacking.mean()) if len(attacking) > 0 else 0.0

    # xT proxy: forward (increasing normalized x) ball progress over the window.
    xt_proxy = 0.0
    if len(balls) > 1 and "frame_idx" in balls.columns:
        ordered = balls.sort_values("frame_idx")
        xs = ((ordered["x1"] + ordered["x2"]) / 2.0 / max(video_width, 1.0)).tolist()
        forward = sum(max(0.0, xs[i + 1] - xs[i]) for i in range(len(xs) - 1))
        xt_proxy = float(min(1.0, forward))

    return LiveMatchState(
        window_start=window_start,
        window_end=window_end,
        n_frames=n_frames,
        n_player=int(len(players)),
        n_ball=int(len(balls)),
        possession_share=possession_share,
        territory=territory,
        pressure_rate=pressure_rate,
        xt_proxy=xt_proxy,
        ball_center_x=ball_center_x,
        ball_center_y=ball_center_y,
    )
