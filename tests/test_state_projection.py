from soccer_edge.video.detector import Detection
from soccer_edge.video.homography import build_homography
from soccer_edge.video.state_projection import detection_to_ball_state, track_to_player_state
from soccer_edge.video.tracker import TrackState


def make_transform():
    return build_homography(
        pixel_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
        pitch_points=[(0, 0), (105, 0), (105, 68), (0, 68)],
    )


def test_detection_to_ball_state() -> None:
    detection = Detection(frame_idx=1, class_name="ball", confidence=0.9, x1=40, y1=40, x2=60, y2=60)
    state = detection_to_ball_state(detection, "video", 1.0, make_transform())
    assert state is not None
    assert round(state.x_m, 6) == 52.5
    assert round(state.y_m, 6) == 34.0


def test_track_to_player_state() -> None:
    track = TrackState(frame_idx=1, track_id="p1", class_name="player", confidence=0.8, x1=40, y1=40, x2=60, y2=60)
    state = track_to_player_state(track, "video", 1.0, make_transform(), team="home")
    assert state is not None
    assert state.team == "home"
    assert round(state.x_m, 6) == 52.5
