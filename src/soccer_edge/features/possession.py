from dataclasses import dataclass

from soccer_edge.features.spatial import PlayerPoint, Point, nearest_players_to_point


@dataclass(frozen=True)
class PossessionEstimate:
    team: str
    player_id: str
    distance_to_ball_m: float
    confidence: float


@dataclass(frozen=True)
class PossessionChain:
    team: str
    start_second: float
    end_second: float

    @property
    def duration_seconds(self) -> float:
        return self.end_second - self.start_second


def estimate_possession(
    players: list[PlayerPoint],
    ball: Point,
    control_radius_m: float = 2.0,
) -> PossessionEstimate | None:
    nearest = nearest_players_to_point(players, ball)
    if not nearest:
        return None
    player, distance = nearest[0]
    if distance > control_radius_m:
        return None
    confidence = max(0.0, 1.0 - distance / control_radius_m)
    return PossessionEstimate(
        team=player.team,
        player_id=player.player_id,
        distance_to_ball_m=distance,
        confidence=confidence,
    )


def build_possession_chains(samples: list[tuple[float, str]]) -> list[PossessionChain]:
    if not samples:
        return []
    ordered = sorted(samples, key=lambda item: item[0])
    chains: list[PossessionChain] = []
    start_second, current_team = ordered[0]
    last_second = start_second

    for timestamp, team in ordered[1:]:
        if team != current_team:
            chains.append(PossessionChain(team=current_team, start_second=start_second, end_second=last_second))
            start_second = timestamp
            current_team = team
        last_second = timestamp

    chains.append(PossessionChain(team=current_team, start_second=start_second, end_second=last_second))
    return chains
