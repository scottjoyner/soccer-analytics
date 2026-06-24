from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soccer_edge.features.spatial import Point


@dataclass(frozen=True)
class HomographyTransform:
    matrix: np.ndarray

    def transform_pixel(self, x_px: float, y_px: float) -> Point | None:
        vector = np.array([x_px, y_px, 1.0], dtype=float)
        transformed = self.matrix @ vector
        if transformed[2] == 0:
            return None
        return Point(x_m=float(transformed[0] / transformed[2]), y_m=float(transformed[1] / transformed[2]))


def build_homography(pixel_points: list[tuple[float, float]], pitch_points: list[tuple[float, float]]) -> HomographyTransform:
    if len(pixel_points) != len(pitch_points):
        raise ValueError("pixel_points and pitch_points must have the same length")
    if len(pixel_points) < 4:
        raise ValueError("At least four point pairs are required")

    rows = []
    for (x, y), (u, v) in zip(pixel_points, pitch_points, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, x * u, y * u, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, x * v, y * v, v])

    _, _, vh = np.linalg.svd(np.array(rows, dtype=float))
    matrix = vh[-1].reshape(3, 3)
    matrix = matrix / matrix[2, 2]
    return HomographyTransform(matrix=matrix)
