from dataclasses import dataclass
from math import hypot

from soccer_edge.features.spatial import PlayerPoint, Point


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


@dataclass(frozen=True)
class GoalGeometry:
    distance_to_goal_m: float
    players_between_ball_and_goal: int


def goal_center(attacking_left_to_right: bool = True) -> Point:
    return Point(PITCH_LENGTH_M if attacking_left_to_right else 0.0, PITCH_WIDTH_M / 2.0)


def distance_to_goal(point: Point, attacking_left_to_right: bool = True) -> float:
    goal = goal_center(attacking_left_to_right)
    return hypot(goal.x_m - point.x_m, goal.y_m - point.y_m)


def players_between_ball_and_goal(
    players: list[PlayerPoint],
    ball: Point,
    attacking_left_to_right: bool = True,
    corridor_width_m: float = 8.0,
) -> int:
    goal = goal_center(attacking_left_to_right)
    dx = goal.x_m - ball.x_m
    dy = goal.y_m - ball.y_m
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return 0

    count = 0
    for player in players:
        px = player.point.x_m - ball.x_m
        py = player.point.y_m - ball.y_m
        projection = (px * dx + py * dy) / length_sq
        if not 0.0 <= projection <= 1.0:
            continue
        closest_x = ball.x_m + projection * dx
        closest_y = ball.y_m + projection * dy
        lateral_distance = hypot(player.point.x_m - closest_x, player.point.y_m - closest_y)
        if lateral_distance <= corridor_width_m / 2.0:
            count += 1
    return count


def goal_geometry(
    ball: Point,
    players: list[PlayerPoint],
    attacking_left_to_right: bool = True,
    corridor_width_m: float = 8.0,
) -> GoalGeometry:
    return GoalGeometry(
        distance_to_goal_m=distance_to_goal(ball, attacking_left_to_right),
        players_between_ball_and_goal=players_between_ball_and_goal(
            players=players,
            ball=ball,
            attacking_left_to_right=attacking_left_to_right,
            corridor_width_m=corridor_width_m,
        ),
    )
