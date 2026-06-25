from pathlib import Path

from soccer_edge.schemas import BallStateRecord
from soccer_edge.video.state_tables import rows_to_dataframe, write_video_state_tables


def test_rows_to_dataframe_from_dataclass() -> None:
    df = rows_to_dataframe([BallStateRecord("video", 1, 1.0, 50.0, 34.0, 0.9)])
    assert list(df.columns) == ["video_id", "frame_idx", "timestamp_seconds", "x_m", "y_m", "confidence", "interpolated"]
    assert df.iloc[0]["video_id"] == "video"


def test_write_video_state_tables(tmp_path: Path) -> None:
    paths = write_video_state_tables(
        tmp_path,
        ball_states=[BallStateRecord("video", 1, 1.0, 50.0, 34.0, 0.9)],
    )
    assert paths["ball_states"].exists()
    assert paths["detections"].exists()
