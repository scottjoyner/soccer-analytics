from dataclasses import dataclass

from soccer_edge.features.spatial import Point


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PENALTY_BOX_DEPTH_M = 16.5
PENALTY_BOX_WIDTH_M = 40.32


@dataclass(frozen=True)
class ZoneFlags:
    in_final_third: bool
    in_penalty_box: bool


def is_in_final_third(point: Point, attacking_left_to_right: bool = True) -> bool:
    if attacking_left_to_right:
        return point.x_m >= PITCH_LENGTH_M * (2.0 / 3.0)
    return point.x_m <= PITCH_LENGTH_M * (1.0 / 3.0)


def is_in_penalty_box(point: Point, attacking_left_to_right: bool = True) -> bool:
    half_box_width = PENALTY_BOX_WIDTH_M / 2.0
    box_y_min = PITCH_WIDTH_M / 2.0 - half_box_width
    box_y_max = PITCH_WIDTH_M / 2.0 + half_box_width
    in_y = box_y_min <= point.y_m <= box_y_max
    if attacking_left_to_right:
        in_x = point.x_m >= PITCH_LENGTH_M - PENALTY_BOX_DEPTH_M
    else:
        in_x = point.x_m <= PENALTY_BOX_DEPTH_M
    return in_x and in_y


def zone_flags(point: Point, attacking_left_to_right: bool = True) -> ZoneFlags:
    return ZoneFlags(
        in_final_third=is_in_final_third(point, attacking_left_to_right),
        in_penalty_box=is_in_penalty_box(point, attacking_left_to_right),
    )


def count_zone_entries(points: list[Point], attacking_left_to_right: bool = True, zone: str = "final_third") -> int:
    if len(points) < 2:
        return 0
    if zone not in {"final_third", "penalty_box"}:
        raise ValueError("zone must be final_third or penalty_box")

    def inside(point: Point) -> bool:
        if zone == "final_third":
            return is_in_final_third(point, attacking_left_to_right)
        return is_in_penalty_box(point, attacking_left_to_right)

    entries = 0
    was_inside = inside(points[0])
    for point in points[1:]:
        now_inside = inside(point)
        if now_inside and not was_inside:
            entries += 1
        was_inside = now_inside
    return entries
