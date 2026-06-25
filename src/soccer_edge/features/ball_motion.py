from dataclasses import dataclass
from math import hypot

from soccer_edge.schemas import BallStateRecord


@dataclass(frozen=True)
class BallMotionRecord:
    video_id: str
    frame_idx: int
    timestamp_seconds: float
    speed_mps: float | None
    acceleration_mps2: float | None


def compute_ball_motion(states: list[BallStateRecord]) -> list[BallMotionRecord]:
    ordered = sorted(states, key=lambda state: state.timestamp_seconds)
    output: list[BallMotionRecord] = []
    previous_speed: float | None = None

    for idx, state in enumerate(ordered):
        speed: float | None = None
        acceleration: float | None = None

        if idx > 0:
            prev = ordered[idx - 1]
            if state.x_m is not None and state.y_m is not None and prev.x_m is not None and prev.y_m is not None:
                dt = state.timestamp_seconds - prev.timestamp_seconds
                if dt > 0:
                    distance = hypot(state.x_m - prev.x_m, state.y_m - prev.y_m)
                    speed = distance / dt
                    if previous_speed is not None:
                        acceleration = (speed - previous_speed) / dt
                    previous_speed = speed

        output.append(
            BallMotionRecord(
                video_id=state.video_id,
                frame_idx=state.frame_idx,
                timestamp_seconds=state.timestamp_seconds,
                speed_mps=speed,
                acceleration_mps2=acceleration,
            )
        )

    return output
