"""Pitch-space feature definitions for soccer video analytics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BallPressureFeature:
    """Basic pressure feature around the ball."""

    nearest_player_distance_to_ball: float | None
    second_nearest_player_distance_to_ball: float | None
    players_within_3m_ball: int
    players_within_5m_ball: int


def empty_ball_pressure_feature() -> BallPressureFeature:
    """Return an empty pressure feature for placeholder pipelines."""

    return BallPressureFeature(
        nearest_player_distance_to_ball=None,
        second_nearest_player_distance_to_ball=None,
        players_within_3m_ball=0,
        players_within_5m_ball=0,
    )
