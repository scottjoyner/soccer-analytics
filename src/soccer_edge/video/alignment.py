from dataclasses import dataclass


@dataclass(frozen=True)
class ClipAlignment:
    clip_id: str
    match_id: str
    video_start_second: float
    match_start_second: float
    period: str = "unknown"

    def video_to_match_second(self, video_second: float) -> float:
        return self.match_start_second + (video_second - self.video_start_second)

    def match_to_video_second(self, match_second: float) -> float:
        return self.video_start_second + (match_second - self.match_start_second)


@dataclass(frozen=True)
class GoalSegment:
    clip_id: str
    match_id: str
    scoring_team: str
    goal_match_second: float
    segment_start_second: float
    segment_end_second: float

    def contains_match_second(self, match_second: float) -> bool:
        return self.segment_start_second <= match_second <= self.segment_end_second
