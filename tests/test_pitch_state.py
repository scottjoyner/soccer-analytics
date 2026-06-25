from soccer_edge.video.detector import Detection
from soccer_edge.video.homography import build_homography
from soccer_edge.video.pitch_state import detection_to_ball_state, track_to_player_state
from soccer_edge.video.tracker import TrackState


def test_detection_to_ball_state() -> None:
    transform = build_homography(
        pixel_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
        pitch_points=[(0, 0), (105, 0), (105, 68), (0, 68)],
    )
    detection = Detection(frame_idx=1, class_name="ball", confidence=0.9, x1=45, y1=45, x2=55, y2=55)
    state = detection_to_ball_state("video", detection, transform, timestamp_seconds=1.0)
    assert state is not None
    assert round(state.x_m, 6) == 52.5
    assert round(state.y_m, 6) == 34.0


def test_track_to_player_state() -> None:
    transform = build_homography(
        pixel_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
        pitch_points=[(0, 0), (105, 0), (105, 68), (0, 68)],
    )
    track = TrackState(frame_idx=2, track_id="p1", class_name="player", confidence=0.8, x1=0, y1=0, x2=10, y2=10)
    state = track_to_player_state("video", track, transform, timestamp_seconds=2.0, team="home")
    assert state is not None
    assert state.team == "home"
    assert state.track_id == "p1"
