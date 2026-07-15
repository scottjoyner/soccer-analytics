import pandas as pd

from soccer_edge.features.table_builders import build_prematch_table
from soccer_edge.player_stats import (
    aggregate_roster_to_team,
    build_player_aggregates,
    build_player_match_stats,
    expected_starter_flag,
    normalize_per_90,
)


def test_normalize_per_90_basic() -> None:
    assert normalize_per_90(2, 45) == 4.0
    assert normalize_per_90(3, 90) == 3.0
    assert normalize_per_90(0, 45) == 0.0


def test_normalize_per_90_guards_zero_minutes() -> None:
    assert normalize_per_90(5, 0) == 0.0
    assert normalize_per_90(5, None) == 0.0


def test_normalize_per_90_series() -> None:
    result = normalize_per_90(pd.Series([2, 3]), pd.Series([45, 90]))
    assert list(result) == [4.0, 3.0]


def test_expected_starter_flag_present() -> None:
    lineup = pd.DataFrame({"player_id": ["A", "B", "C"]})
    assert expected_starter_flag(lineup, "B") == 1.0


def test_expected_starter_flag_absent() -> None:
    lineup = pd.DataFrame({"player_id": ["A", "B", "C"]})
    assert expected_starter_flag(lineup, "Z") == 0.0


def test_expected_starter_flag_missing_lineup() -> None:
    assert expected_starter_flag(None, "A") == 0.5
    assert expected_starter_flag(pd.DataFrame(), "A") == 0.5
    assert expected_starter_flag(pd.DataFrame({"other": [1]}), "A", player_column="player_id") == 0.5


def test_player_match_stats_adds_per_90_and_starter() -> None:
    events = pd.DataFrame(
        [
            {"match_id": 1, "player": {"name": "A"}, "team": {"name": "Home"}, "type": {"name": "Shot"}, "shot": {"outcome": {"name": "Goal"}}, "minute": 45},
            {"match_id": 1, "player": {"name": "A"}, "team": {"name": "Home"}, "type": {"name": "Pass"}, "pass": {}, "minute": 46},
        ]
    )
    lineup = pd.DataFrame({"player_id": ["A"]})
    stats = build_player_match_stats(events, lineup=lineup)
    row = stats[stats["player_name"] == "A"].iloc[0]
    assert round(row["goals_per_90"], 4) == round(1 / 46 * 90, 4)
    assert round(row["shots_per_90"], 4) == round(1 / 46 * 90, 4)
    assert row["is_expected_starter"] == 1.0


def test_player_aggregates_per_90() -> None:
    stats = pd.DataFrame(
        [
            {"match_id": 1, "player_name": "A", "team_name": "Home", "goals": 1, "shots": 2, "max_minute": 45},
            {"match_id": 2, "player_name": "A", "team_name": "Home", "goals": 1, "shots": 0, "max_minute": 90},
        ]
    )
    agg = build_player_aggregates(stats)
    a = agg[agg["player_name"] == "A"].iloc[0]
    assert a["total_minutes"] == 135
    assert round(a["goals_per_90"], 4) == round(2 / 135 * 90, 4)


def test_aggregate_roster_to_team() -> None:
    players = pd.DataFrame(
        [
            {"team_name": "Home", "goals_per_90": 1.0, "shots_per_90": 3.0, "is_expected_starter": 1.0},
            {"team_name": "Home", "goals_per_90": 2.0, "shots_per_90": 1.0, "is_expected_starter": 0.0},
            {"team_name": "Away", "goals_per_90": 0.5, "shots_per_90": 4.0, "is_expected_starter": 1.0},
        ]
    )
    roster = aggregate_roster_to_team(players)
    assert set(roster["team_name"]) == {"Home", "Away"}
    home = roster[roster["team_name"] == "Home"].iloc[0]
    assert home["goals_per_90_mean"] == 1.5
    assert home["goals_per_90_min"] == 1.0
    assert home["goals_per_90_max"] == 2.0
    assert home["expected_starter_count"] == 1
    away = roster[roster["team_name"] == "Away"].iloc[0]
    assert away["expected_starter_count"] == 1


def test_prematch_table_adds_team_roster_columns() -> None:
    matches = pd.DataFrame([{"match_id": "m1", "home_team": "Home", "away_team": "Away"}])
    player_features = pd.DataFrame(
        [
            {"team_name": "Home", "goals_per_90": 1.0, "is_expected_starter": 1.0},
            {"team_name": "Home", "goals_per_90": 2.0, "is_expected_starter": 0.0},
            {"team_name": "Away", "goals_per_90": 0.5, "is_expected_starter": 1.0},
        ]
    )
    table = build_prematch_table(matches, player_features=player_features)
    assert table.iloc[0]["home_goals_per_90_mean"] == 1.5
    assert table.iloc[0]["home_expected_starter_count"] == 1
    assert table.iloc[0]["away_expected_starter_count"] == 1
