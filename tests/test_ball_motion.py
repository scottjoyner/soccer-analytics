from soccer_edge.features.ball_motion import compute_ball_motion
from soccer_edge.schemas import BallStateRecord


def test_compute_ball_motion_speed_and_acceleration() -> None:
    states = [
        BallStateRecord("video", 0, 0.0, 0.0, 0.0, 1.0),
        BallStateRecord("video", 1, 1.0, 3.0, 4.0, 1.0),
        BallStateRecord("video", 2, 2.0, 9.0, 12.0, 1.0),
    ]
    motion = compute_ball_motion(states)
    assert motion[0].speed_mps is None
    assert motion[1].speed_mps == 5.0
    assert motion[2].speed_mps == 10.0
    assert motion[2].acceleration_mps2 == 5.0
