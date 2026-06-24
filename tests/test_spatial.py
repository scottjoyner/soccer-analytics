from soccer_edge.features.spatial import Point, PlayerPoint, count_players_within_radius, nearest_players_to_point


def test_nearest_players_to_point() -> None:
    ball = Point(50.0, 34.0)
    players = [
        PlayerPoint("p1", "A", Point(51.0, 34.0)),
        PlayerPoint("p2", "B", Point(60.0, 34.0)),
    ]
    nearest = nearest_players_to_point(players, ball)
    assert nearest[0][0].player_id == "p1"
    assert nearest[0][1] == 1.0


def test_count_players_within_radius() -> None:
    ball = Point(50.0, 34.0)
    players = [
        PlayerPoint("p1", "A", Point(51.0, 34.0)),
        PlayerPoint("p2", "B", Point(56.0, 34.0)),
    ]
    assert count_players_within_radius(players, ball, 3.0) == 1
    assert count_players_within_radius(players, ball, 6.0) == 2
