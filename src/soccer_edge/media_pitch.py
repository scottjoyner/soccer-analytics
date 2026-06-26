from dataclasses import asdict, is_dataclass
from typing import Iterable

from soccer_edge.video.homography import HomographyTransform


def row_mapping(row: object) -> dict[str, object]:
    if is_dataclass(row):
        return asdict(row)
    if isinstance(row, dict):
        return dict(row)
    return dict(vars(row))


def add_pitch_point(row: object, transform: HomographyTransform) -> dict[str, object]:
    output = row_mapping(row)
    x1 = float(output.get("x1", 0.0))
    y1 = float(output.get("y1", 0.0))
    x2 = float(output.get("x2", 0.0))
    y2 = float(output.get("y2", 0.0))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    point = transform.transform_pixel(center_x, center_y)
    output["pixel_center_x"] = center_x
    output["pixel_center_y"] = center_y
    if point is not None:
        output["pitch_x_m"] = point.x_m
        output["pitch_y_m"] = point.y_m
    return output


def add_pitch_points(rows: Iterable[object], transform: HomographyTransform | None = None) -> list[object]:
    if transform is None:
        return list(rows)
    return [add_pitch_point(row, transform) for row in rows]
