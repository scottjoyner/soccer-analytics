from soccer_edge.features.goal_geometry import distance_to_goal, goal_geometry, players_between_ball_and_goal
from soccer_edge.features.spatial import PlayerPoint, Point


def test_distance_to_goal() -> None:
    assert distance_to_goal(Point(105.0, 34.0), attacking_left_to_right=True) == 0.0
    assert distance_to_goal(Point(0.0, 34.0), attacking_left_to_right=False) == 0.0


def test_players_between_ball_and_goal() -> None:
    ball = Point(50.0, 34.0)
    players = [
        PlayerPoint("p1", "away", Point(70.0, 34.0)),
        PlayerPoint("p2", "away", Point(70.0, 50.0)),
    ]
    assert players_between_ball_and_goal(players, ball, corridor_width_m=8.0) == 1


def test_goal_geometry() -> None:
    ball = Point(100.0, 34.0)
    players = [PlayerPoint("p1", "away", Point(102.0, 34.0))]
    geometry = goal_geometry(ball, players)
    assert geometry.distance_to_goal_m == 5.0
    assert geometry.players_between_ball_and_goal == 1
