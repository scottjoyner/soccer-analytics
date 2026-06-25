from soccer_edge.features.spatial import PlayerPoint, Point
from soccer_edge.features.team_shape import compute_line_heights, compute_team_shape


def test_compute_team_shape() -> None:
    players = [
        PlayerPoint("p1", "home", Point(0.0, 0.0)),
        PlayerPoint("p2", "home", Point(2.0, 0.0)),
    ]
    shape = compute_team_shape(players, "home")
    assert shape is not None
    assert shape.centroid_x_m == 1.0
    assert shape.centroid_y_m == 0.0
    assert shape.compactness_m == 1.0


def test_compute_line_heights() -> None:
    players = [
        PlayerPoint("p1", "home", Point(10.0, 0.0)),
        PlayerPoint("p2", "home", Point(80.0, 0.0)),
    ]
    lines = compute_line_heights(players, "home", attacking_left_to_right=True)
    assert lines is not None
    assert lines.defensive_line_x_m == 10.0
    assert lines.attacking_line_x_m == 80.0
