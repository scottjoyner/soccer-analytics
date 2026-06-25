from dataclasses import dataclass
from math import hypot

from soccer_edge.schemas import BallStateRecord


@dataclass(frozen=True)
class BallSpeedSample:
    frame_idx: int
    timestamp_seconds: float
    speed_mps: float


@dataclass(frozen=True)
class BallAccelerationSample:
    frame_idx: int
    timestamp_seconds: float
    acceleration_mps2: float


def compute_ball_speeds(states: list[BallStateRecord]) -> list[BallSpeedSample]:
    ordered = [state for state in sorted(states, key=lambda item: item.timestamp_seconds) if state.x_m is not None and state.y_m is not None]
    samples: list[BallSpeedSample] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        dt = current.timestamp_seconds - previous.timestamp_seconds
        if dt <= 0:
            continue
        distance = hypot(current.x_m - previous.x_m, current.y_m - previous.y_m)
        samples.append(BallSpeedSample(current.frame_idx, current.timestamp_seconds, distance / dt))
    return samples


def compute_ball_accelerations(speeds: list[BallSpeedSample]) -> list[BallAccelerationSample]:
    ordered = sorted(speeds, key=lambda item: item.timestamp_seconds)
    samples: list[BallAccelerationSample] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        dt = current.timestamp_seconds - previous.timestamp_seconds
        if dt <= 0:
            continue
        samples.append(
            BallAccelerationSample(
                frame_idx=current.frame_idx,
                timestamp_seconds=current.timestamp_seconds,
                acceleration_mps2=(current.speed_mps - previous.speed_mps) / dt,
            )
        )
    return samples
