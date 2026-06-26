import pandas as pd
import pytest

from soccer_edge.features.table_builders import build_inplay_rolling_table, build_prematch_table


def test_build_prematch_table() -> None:
    matches = pd.DataFrame([{"match_id": "m1", "home_team": "A", "away_team": "B"}])
    table = build_prematch_table(matches)
    assert table.iloc[0]["match_id"] == "m1"


def test_build_inplay_rolling_table() -> None:
    frame = pd.DataFrame(
        [
            {"match_id": "m1", "timestamp_seconds": 1.0, "speed": 1.0},
            {"match_id": "m1", "timestamp_seconds": 2.0, "speed": 3.0},
            {"match_id": "m1", "timestamp_seconds": 5.0, "speed": 10.0},
        ]
    )
    table = build_inplay_rolling_table(frame, feature_columns=["speed"], window_seconds=2.0)
    assert list(table["speed_last"]) == [1.0, 3.0, 10.0]
    assert table.iloc[1]["speed_mean_2s"] == 2.0


def test_build_inplay_rolling_table_carries_metadata() -> None:
    frame = pd.DataFrame(
        [
            {"match_id": "m1", "timestamp_seconds": 1.0, "speed": 1.0, "source_name": "demo"},
            {"match_id": "m1", "timestamp_seconds": 2.0, "speed": 3.0, "source_name": "demo"},
        ]
    )
    table = build_inplay_rolling_table(frame, feature_columns=["speed"], window_seconds=2.0, carry_columns=["source_name"])
    assert list(table["source_name"]) == ["demo", "demo"]


def test_build_inplay_rolling_table_requires_positive_window() -> None:
    with pytest.raises(ValueError):
        build_inplay_rolling_table(pd.DataFrame(), feature_columns=[], window_seconds=0.0)
