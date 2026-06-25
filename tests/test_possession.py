from soccer_edge.features.possession import build_possession_chains, estimate_possession
from soccer_edge.features.spatial import PlayerPoint, Point


def test_estimate_possession_near_ball() -> None:
    players = [
        PlayerPoint("p1", "home", Point(50.5, 34.0)),
        PlayerPoint("p2", "away", Point(60.0, 34.0)),
    ]
    estimate = estimate_possession(players, Point(50.0, 34.0), control_radius_m=2.0)
    assert estimate is not None
    assert estimate.team == "home"
    assert estimate.confidence == 0.75


def test_build_possession_chains() -> None:
    chains = build_possession_chains([(0.0, "home"), (1.0, "home"), (2.0, "away"), (3.0, "away")])
    assert len(chains) == 2
    assert chains[0].team == "home"
    assert chains[0].duration_seconds == 1.0
    assert chains[1].team == "away"
