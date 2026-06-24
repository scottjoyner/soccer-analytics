from soccer_edge.schemas import BallStateRecord


def interpolate_ball_states(states: list[BallStateRecord]) -> list[BallStateRecord]:
    if not states:
        return []

    ordered = sorted(states, key=lambda state: state.frame_idx)
    by_frame = {state.frame_idx: state for state in ordered}
    min_frame = ordered[0].frame_idx
    max_frame = ordered[-1].frame_idx
    output: list[BallStateRecord] = []

    known_frames = [state.frame_idx for state in ordered if state.x_m is not None and state.y_m is not None]

    for frame_idx in range(min_frame, max_frame + 1):
        existing = by_frame.get(frame_idx)
        if existing and existing.x_m is not None and existing.y_m is not None:
            output.append(existing)
            continue

        before = max((frame for frame in known_frames if frame < frame_idx), default=None)
        after = min((frame for frame in known_frames if frame > frame_idx), default=None)

        if before is None or after is None:
            if existing:
                output.append(existing)
            continue

        start = by_frame[before]
        end = by_frame[after]
        ratio = (frame_idx - before) / (after - before)
        x_m = start.x_m + (end.x_m - start.x_m) * ratio
        y_m = start.y_m + (end.y_m - start.y_m) * ratio
        timestamp = existing.timestamp_seconds if existing else start.timestamp_seconds + ratio * (end.timestamp_seconds - start.timestamp_seconds)

        output.append(
            BallStateRecord(
                video_id=start.video_id,
                frame_idx=frame_idx,
                timestamp_seconds=timestamp,
                x_m=x_m,
                y_m=y_m,
                confidence=min(start.confidence, end.confidence) * 0.5,
                interpolated=True,
            )
        )

    return output
