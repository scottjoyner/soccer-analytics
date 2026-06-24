from soccer_edge.schemas import BallStateRecord
from soccer_edge.video.ball_interpolation import interpolate_ball_states


def test_interpolate_ball_states_fills_middle_frame() -> None:
    states = [
        BallStateRecord("video", 0, 0.0, 0.0, 0.0, 0.9),
        BallStateRecord("video", 2, 2.0, 2.0, 2.0, 0.8),
    ]

    output = interpolate_ball_states(states)
    middle = output[1]
    assert middle.frame_idx == 1
    assert middle.x_m == 1.0
    assert middle.y_m == 1.0
    assert middle.interpolated is True
    assert middle.confidence == 0.4


def test_interpolate_ball_states_keeps_known_frames() -> None:
    states = [BallStateRecord("video", 0, 0.0, 5.0, 6.0, 0.9)]
    output = interpolate_ball_states(states)
    assert output == states
