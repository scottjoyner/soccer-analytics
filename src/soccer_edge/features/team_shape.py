from dataclasses import dataclass

from soccer_edge.features.spatial import PlayerPoint


@dataclass(frozen=True)
class TeamShape:
    team: str
    centroid_x_m: float
    centroid_y_m: float
    compactness_m: float
    min_x_m: float
    max_x_m: float


@dataclass(frozen=True)
class LineHeights:
    team: str
    defensive_line_x_m: float
    attacking_line_x_m: float


def team_players(players: list[PlayerPoint], team: str) -> list[PlayerPoint]:
    return [player for player in players if player.team == team]


def compute_team_shape(players: list[PlayerPoint], team: str) -> TeamShape | None:
    selected = team_players(players, team)
    if not selected:
        return None
    xs = [player.point.x_m for player in selected]
    ys = [player.point.y_m for player in selected]
    centroid_x = sum(xs) / len(xs)
    centroid_y = sum(ys) / len(ys)
    compactness = sum(((x - centroid_x) ** 2 + (y - centroid_y) ** 2) ** 0.5 for x, y in zip(xs, ys, strict=True)) / len(selected)
    return TeamShape(
        team=team,
        centroid_x_m=centroid_x,
        centroid_y_m=centroid_y,
        compactness_m=compactness,
        min_x_m=min(xs),
        max_x_m=max(xs),
    )


def compute_line_heights(players: list[PlayerPoint], team: str, attacking_left_to_right: bool = True) -> LineHeights | None:
    selected = team_players(players, team)
    if not selected:
        return None
    xs = sorted(player.point.x_m for player in selected)
    if attacking_left_to_right:
        defensive = xs[0]
        attacking = xs[-1]
    else:
        defensive = xs[-1]
        attacking = xs[0]
    return LineHeights(team=team, defensive_line_x_m=defensive, attacking_line_x_m=attacking)
