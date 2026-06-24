from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Point:
    x_m: float
    y_m: float


@dataclass(frozen=True)
class PlayerPoint:
    player_id: str
    team: str
    point: Point


def distance_m(a: Point, b: Point) -> float:
    return hypot(a.x_m - b.x_m, a.y_m - b.y_m)


def nearest_players_to_point(players: list[PlayerPoint], target: Point) -> list[tuple[PlayerPoint, float]]:
    distances = [(player, distance_m(player.point, target)) for player in players]
    return sorted(distances, key=lambda item: item[1])


def count_players_within_radius(players: list[PlayerPoint], target: Point, radius_m: float) -> int:
    return sum(1 for player in players if distance_m(player.point, target) <= radius_m)
