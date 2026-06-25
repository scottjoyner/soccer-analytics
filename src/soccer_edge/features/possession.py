from dataclasses import dataclass

from soccer_edge.features.spatial import PlayerPoint, Point, distance_m, nearest_players_to_point


@dataclass(frozen=True)
class PossessionEstimate:
    timestamp_seconds: float
    team: str
    player_id: str
    distance_to_ball_m: float
    confidence: float


@dataclass(frozen=True)
class PossessionChain:
    team: str
    start_second: float
    end_second: float
    frame_count: int = 1

    @property
    def duration_seconds(self) -> float:
        return self.end_second - self.start_second


def estimate_possession(
    players: list[PlayerPoint],
    ball: Point,
    control_radius_m: float = 3.0,
    timestamp_seconds: float = 0.0,
) -> PossessionEstimate | None:
    nearest = nearest_players_to_point(players, ball)
    if not nearest:
        return None
    player, distance = nearest[0]
    if distance > control_radius_m:
        return None
    confidence = max(0.0, 1.0 - distance / control_radius_m)
    return PossessionEstimate(
        timestamp_seconds=timestamp_seconds,
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
    frame_count = 1

    for timestamp, team in ordered[1:]:
        if team != current_team:
            chains.append(PossessionChain(team=current_team, start_second=start_second, end_second=last_second, frame_count=frame_count))
            start_second = timestamp
            current_team = team
            frame_count = 1
        else:
            frame_count += 1
        last_second = timestamp

    chains.append(PossessionChain(team=current_team, start_second=start_second, end_second=last_second, frame_count=frame_count))
    return chains


def build_chains_from_estimates(estimates: list[PossessionEstimate], min_confidence: float = 0.25) -> list[PossessionChain]:
    samples = [(estimate.timestamp_seconds, estimate.team) for estimate in estimates if estimate.confidence >= min_confidence]
    return build_possession_chains(samples)


def possession_pressure_score(players: list[PlayerPoint], ball: Point, defending_team: str, radius_m: float = 5.0) -> float:
    defenders = [player for player in players if player.team == defending_team]
    return sum(max(0.0, radius_m - distance_m(player.point, ball)) for player in defenders)
