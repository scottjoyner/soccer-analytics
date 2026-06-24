from soccer_edge.schemas import BallStateRecord, MatchRecord


def test_match_record_defaults() -> None:
    record = MatchRecord(
        match_id="match_1",
        competition="cup",
        season="2026",
        match_date="2026-06-11",
        home_team="A",
        away_team="B",
    )
    assert record.status == "scheduled"


def test_ball_state_interpolation_flag() -> None:
    record = BallStateRecord(
        video_id="video_1",
        frame_idx=1,
        timestamp_seconds=1.0,
        x_m=50.0,
        y_m=34.0,
        confidence=0.5,
        interpolated=True,
    )
    assert record.interpolated is True
