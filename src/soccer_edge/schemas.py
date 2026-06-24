from dataclasses import dataclass


@dataclass(frozen=True)
class MatchRecord:
    match_id: str
    competition: str
    season: str
    match_date: str
    home_team: str
    away_team: str
    stage: str = ""
    status: str = "scheduled"


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    match_id: str
    period: int
    timestamp_seconds: float
    team: str
    player: str
    event_type: str
    x_m: float | None = None
    y_m: float | None = None
    outcome: str = ""


@dataclass(frozen=True)
class FrameRecord:
    video_id: str
    match_id: str
    frame_idx: int
    video_second: float
    match_second: float | None = None
    period: str = ""


@dataclass(frozen=True)
class BallStateRecord:
    video_id: str
    frame_idx: int
    timestamp_seconds: float
    x_m: float | None
    y_m: float | None
    confidence: float
    interpolated: bool = False


@dataclass(frozen=True)
class PlayerStateRecord:
    video_id: str
    frame_idx: int
    timestamp_seconds: float
    track_id: str
    team: str
    x_m: float | None
    y_m: float | None
    confidence: float


@dataclass(frozen=True)
class FeatureRecord:
    match_id: str
    timestamp_seconds: float
    feature_name: str
    value: float


@dataclass(frozen=True)
class LabelRecord:
    match_id: str
    feature_timestamp_seconds: float
    label_name: str
    label_value: int
    label_window_seconds: float | None = None
