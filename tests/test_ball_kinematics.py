from soccer_edge.features.ball_kinematics import compute_ball_accelerations, compute_ball_speeds
from soccer_edge.schemas import BallStateRecord


def test_compute_ball_speeds() -> None:
    states = [
        BallStateRecord("video", 0, 0.0, 0.0, 0.0, 1.0),
        BallStateRecord("video", 1, 2.0, 6.0, 8.0, 1.0),
    ]
    speeds = compute_ball_speeds(states)
    assert len(speeds) == 1
    assert speeds[0].speed_mps == 5.0


def test_compute_ball_accelerations() -> None:
    states = [
        BallStateRecord("video", 0, 0.0, 0.0, 0.0, 1.0),
        BallStateRecord("video", 1, 1.0, 1.0, 0.0, 1.0),
        BallStateRecord("video", 2, 2.0, 4.0, 0.0, 1.0),
    ]
    accelerations = compute_ball_accelerations(compute_ball_speeds(states))
    assert len(accelerations) == 1
    assert accelerations[0].acceleration_mps2 == 2.0
