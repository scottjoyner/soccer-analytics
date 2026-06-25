from soccer_edge.features.possession import (
    build_chains_from_estimates,
    build_possession_chains,
    estimate_possession,
    possession_pressure_score,
)
from soccer_edge.features.spatial import PlayerPoint, Point


def test_estimate_possession_near_ball() -> None:
    players = [
        PlayerPoint("p1", "home", Point(50.5, 34.0)),
        PlayerPoint("p2", "away", Point(60.0, 34.0)),
    ]
    estimate = estimate_possession(players, Point(50.0, 34.0), control_radius_m=2.0, timestamp_seconds=12.0)
    assert estimate is not None
    assert estimate.team == "home"
    assert estimate.confidence == 0.75
    assert estimate.timestamp_seconds == 12.0


def test_build_possession_chains() -> None:
    chains = build_possession_chains([(0.0, "home"), (1.0, "home"), (2.0, "away"), (3.0, "away")])
    assert len(chains) == 2
    assert chains[0].team == "home"
    assert chains[0].duration_seconds == 1.0
    assert chains[0].frame_count == 2
    assert chains[1].team == "away"


def test_build_chains_from_estimates() -> None:
    players = [PlayerPoint("p1", "home", Point(50.0, 34.0))]
    first = estimate_possession(players, Point(50.5, 34.0), timestamp_seconds=1.0)
    second = estimate_possession(players, Point(50.6, 34.0), timestamp_seconds=2.0)
    assert first is not None and second is not None
    chains = build_chains_from_estimates([first, second])
    assert len(chains) == 1
    assert chains[0].frame_count == 2


def test_possession_pressure_score() -> None:
    players = [PlayerPoint("a1", "away", Point(52.0, 34.0))]
    assert possession_pressure_score(players, Point(50.0, 34.0), defending_team="away", radius_m=5.0) == 3.0
