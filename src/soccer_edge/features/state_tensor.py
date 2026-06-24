from __future__ import annotations

import numpy as np

from soccer_edge.features.spatial import PlayerPoint, Point


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def point_to_grid_index(point: Point, height_bins: int, width_bins: int) -> tuple[int, int]:
    x_ratio = min(max(point.x_m / PITCH_LENGTH_M, 0.0), 0.999999)
    y_ratio = min(max(point.y_m / PITCH_WIDTH_M, 0.0), 0.999999)
    row = int(y_ratio * height_bins)
    col = int(x_ratio * width_bins)
    return row, col


def build_occupancy_grid(
    players: list[PlayerPoint],
    ball: Point | None,
    height_bins: int = 32,
    width_bins: int = 48,
) -> np.ndarray:
    grid = np.zeros((3, height_bins, width_bins), dtype=np.float32)
    for player in players:
        row, col = point_to_grid_index(player.point, height_bins, width_bins)
        channel = 0 if player.team == "home" else 1
        grid[channel, row, col] += 1.0
    if ball is not None:
        row, col = point_to_grid_index(ball, height_bins, width_bins)
        grid[2, row, col] = 1.0
    return grid
