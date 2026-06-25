from soccer_edge.schemas import BallStateRecord, PlayerStateRecord
from soccer_edge.video.detector import Detection
from soccer_edge.video.homography import HomographyTransform
from soccer_edge.video.tracker import TrackState


def detection_to_ball_state(
    detection: Detection,
    video_id: str,
    timestamp_seconds: float,
    transform: HomographyTransform,
) -> BallStateRecord | None:
    if detection.class_name.lower() != "ball":
        return None
    center_x, center_y = detection.center
    point = transform.transform_pixel(center_x, center_y)
    if point is None:
        return None
    return BallStateRecord(
        video_id=video_id,
        frame_idx=detection.frame_idx,
        timestamp_seconds=timestamp_seconds,
        x_m=point.x_m,
        y_m=point.y_m,
        confidence=detection.confidence,
        interpolated=False,
    )


def track_to_player_state(
    track: TrackState,
    video_id: str,
    timestamp_seconds: float,
    transform: HomographyTransform,
    team: str = "unknown",
) -> PlayerStateRecord | None:
    if track.class_name.lower() != "player":
        return None
    center_x = (track.x1 + track.x2) / 2.0
    center_y = (track.y1 + track.y2) / 2.0
    point = transform.transform_pixel(center_x, center_y)
    if point is None:
        return None
    return PlayerStateRecord(
        video_id=video_id,
        frame_idx=track.frame_idx,
        timestamp_seconds=timestamp_seconds,
        track_id=track.track_id,
        team=team,
        x_m=point.x_m,
        y_m=point.y_m,
        confidence=track.confidence,
    )
