import pandas as pd
import pytest

from soccer_edge.player_stats import build_player_form_features, build_player_match_stats, build_team_player_feature_table, write_player_match_stats


def test_build_player_match_stats_from_statsbomb_like_rows() -> None:
    events = pd.DataFrame(
        [
            {"match_id": 1, "player": {"name": "A"}, "team": {"name": "Home"}, "type": {"name": "Shot"}, "shot": {"outcome": {"name": "Goal"}}, "minute": 10},
            {"match_id": 1, "player": {"name": "A"}, "team": {"name": "Home"}, "type": {"name": "Pass"}, "pass": {}, "minute": 11},
            {"match_id": 1, "player": {"name": "B"}, "team": {"name": "Away"}, "type": {"name": "Pressure"}, "minute": 12},
        ]
    )
    stats = build_player_match_stats(events)
    player_a = stats[stats["player_name"] == "A"].iloc[0]
    assert player_a["shots"] == 1
    assert player_a["goals"] == 1
    assert player_a["completed_passes"] == 1


def test_build_player_form_features() -> None:
    stats = pd.DataFrame(
        [
            {"match_id": 1, "player_name": "A", "team_name": "Home", "shots": 1},
            {"match_id": 2, "player_name": "A", "team_name": "Home", "shots": 3},
        ]
    )
    form = build_player_form_features(stats, window=2)
    assert form.iloc[0]["shots_form_2"] == 0
    assert form.iloc[1]["shots_form_2"] == 1


def test_build_team_player_feature_table() -> None:
    stats = pd.DataFrame(
        [
            {"match_id": 1, "player_name": "A", "team_name": "Home", "shots": 1},
            {"match_id": 1, "player_name": "B", "team_name": "Home", "shots": 2},
        ]
    )
    table = build_team_player_feature_table(stats)
    assert table.iloc[0]["shots"] == 3
    assert table.iloc[0]["player_count"] == 2


def test_write_player_match_stats(tmp_path) -> None:
    source = tmp_path / "events.csv"
    output = tmp_path / "players.csv"
    pd.DataFrame([{"match_id": 1, "player_name": "A", "team_name": "Home", "event_type": "Shot", "shot_outcome": "Goal"}]).to_csv(source, index=False)
    path = write_player_match_stats(source, output)
    assert path.exists()


def test_build_player_form_features_validates_columns() -> None:
    with pytest.raises(ValueError):
        build_player_form_features(pd.DataFrame([{"x": 1}]))
