from soccer_edge.features.spatial import PlayerPoint, Point
from soccer_edge.features.state_tensor import build_occupancy_grid, point_to_grid_index


def test_point_to_grid_index_center() -> None:
    row, col = point_to_grid_index(Point(52.5, 34.0), height_bins=34, width_bins=105)
    assert row == 17
    assert col == 52


def test_build_occupancy_grid() -> None:
    players = [
        PlayerPoint("home_1", "home", Point(10.0, 10.0)),
        PlayerPoint("away_1", "away", Point(20.0, 20.0)),
    ]
    grid = build_occupancy_grid(players, ball=Point(30.0, 30.0), height_bins=10, width_bins=10)
    assert grid.shape == (3, 10, 10)
    assert grid[0].sum() == 1.0
    assert grid[1].sum() == 1.0
    assert grid[2].sum() == 1.0
