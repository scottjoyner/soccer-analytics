import pandas as pd
import pytest

from soccer_edge.player_stats import (
    assign_match_opponents,
    build_player_aggregates,
    build_player_aggregates_from_events,
    build_player_form_features,
    build_player_match_stats,
    build_team_player_feature_table,
    normalize_lineup_players,
    write_player_match_stats,
)


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
    assert player_a["shots_per_90"] == 0
    assert player_a["shots_per_observed90"] > 0


def test_build_player_match_stats_uses_real_minutes_for_per_90() -> None:
    events = pd.DataFrame(
        [
            {"match_id": 1, "player_id": 7, "player_name": "A", "team_name": "Home", "event_type": "Shot", "shot_outcome": "Goal", "minutes_played": 45},
        ]
    )
    stats = build_player_match_stats(events)
    player_a = stats.iloc[0]
    assert player_a["minutes_played"] == 45
    assert player_a["shots_per_90"] == 2


def test_lineup_positions_set_starter_and_minutes() -> None:
    lineup = pd.DataFrame(
        [
            {
                "team_name": "Home",
                "lineup": [
                    {"player_id": 7, "player_name": "A", "positions": [{"from": "00:00", "to": "45:00"}]},
                    {"player_id": 8, "player_name": "B", "positions": [{"from": "60:00", "to": "90:00"}]},
                ],
            }
        ]
    )
    players = normalize_lineup_players(lineup)
    assert players[players["player_id"] == 7]["is_expected_starter"].iloc[0] == 1
    assert players[players["player_id"] == 7]["minutes_played"].iloc[0] == 45
    assert players[players["player_id"] == 8]["is_expected_starter"].iloc[0] == 0


def test_build_player_match_stats_uses_lineup_minutes() -> None:
    events = pd.DataFrame(
        [
            {"match_id": 1, "player": {"id": 7, "name": "A"}, "team": {"name": "Home"}, "type": {"name": "Shot"}, "minute": 10},
        ]
    )
    lineup = pd.DataFrame(
        [
            {
                "team_name": "Home",
                "lineup": [
                    {"player_id": 7, "player_name": "A", "positions": [{"from": "00:00", "to": "90:00"}]},
                ],
            }
        ]
    )
    stats = build_player_match_stats(events, lineup=lineup)
    row = stats.iloc[0]
    assert row["is_expected_starter"] == 1
    assert row["minutes_played"] == 90
    assert row["shots_per_90"] == 1


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


def test_build_player_aggregates() -> None:
    stats = pd.DataFrame(
        [
            {"match_id": 1, "player_name": "A", "team_name": "Home", "total_events": 5, "shots": 2, "goals": 1, "passes": 10, "completed_passes": 8, "carries": 1, "dribbles": 0, "pressures": 1, "interceptions": 0, "tackles": 0, "fouls_committed": 0},
            {"match_id": 2, "player_name": "A", "team_name": "Home", "total_events": 6, "shots": 0, "goals": 0, "passes": 12, "completed_passes": 9, "carries": 2, "dribbles": 1, "pressures": 0, "interceptions": 1, "tackles": 0, "fouls_committed": 1},
            {"match_id": 1, "player_name": "B", "team_name": "Away", "total_events": 4, "shots": 1, "goals": 0, "passes": 5, "completed_passes": 3, "carries": 0, "dribbles": 0, "pressures": 2, "interceptions": 0, "tackles": 1, "fouls_committed": 0},
        ]
    )
    agg = build_player_aggregates(stats)
    a = agg[agg["player_name"] == "A"].iloc[0]
    assert a["appearances"] == 2
    assert a["total_shots"] == 2
    assert a["avg_shots"] == 1.0
    assert a["total_goals"] == 1
    assert a["total_passes"] == 22
    assert a["total_completed_passes"] == 17
    assert round(a["pass_completion_rate"], 4) == round(17 / 22, 4)
    assert "Home" in a["teams"]
    b = agg[agg["player_name"] == "B"].iloc[0]
    assert b["appearances"] == 1


def test_assign_match_opponents() -> None:
    stats = pd.DataFrame(
        [
            {"match_id": 1, "player_name": "A", "team_name": "Home"},
            {"match_id": 1, "player_name": "B", "team_name": "Away"},
            {"match_id": 2, "player_name": "A", "team_name": "Home"},
            {"match_id": 2, "player_name": "B", "team_name": "Away"},
        ]
    )
    opponents = assign_match_opponents(stats)
    assert (opponents[opponents["player_name"] == "A"]["opponent_team"] == "Away").all()
    assert (opponents[opponents["player_name"] == "B"]["opponent_team"] == "Home").all()


def test_build_player_aggregates_split_by_opponent() -> None:
    stats = pd.DataFrame(
        [
            {"match_id": 1, "player_name": "A", "team_name": "Home", "goals": 1, "passes": 5, "completed_passes": 4},
            {"match_id": 2, "player_name": "A", "team_name": "Home", "goals": 0, "passes": 5, "completed_passes": 3},
            {"match_id": 1, "player_name": "B", "team_name": "Away", "goals": 0, "passes": 2, "completed_passes": 1},
            {"match_id": 2, "player_name": "B", "team_name": "Away", "goals": 0, "passes": 2, "completed_passes": 1},
        ]
    )
    agg = build_player_aggregates(stats, split_by=["opponent"])
    a_away = agg[(agg["player_name"] == "A") & (agg["opponent_team"] == "Away")].iloc[0]
    assert a_away["appearances"] == 2
    assert a_away["total_goals"] == 1
    assert "team" in agg.columns
    b_home = agg[(agg["player_name"] == "B") & (agg["opponent_team"] == "Home")].iloc[0]
    assert b_home["appearances"] == 2


def test_build_player_aggregates_split_by_team() -> None:
    stats = pd.DataFrame(
        [
            {"match_id": 1, "player_name": "A", "team_name": "Home", "goals": 1},
            {"match_id": 2, "player_name": "A", "team_name": "Away", "goals": 2},
        ]
    )
    agg = build_player_aggregates(stats, split_by=["team"])
    assert set(agg["team_name"]) == {"Home", "Away"}
    assert agg[agg["team_name"] == "Home"]["total_goals"].iloc[0] == 1
    assert agg[agg["team_name"] == "Away"]["total_goals"].iloc[0] == 2


def test_build_player_aggregates_from_events() -> None:
    match_one = pd.DataFrame(
        [
            {"match_id": 1, "player": {"name": "A"}, "team": {"name": "Home"}, "type": {"name": "Shot"}, "shot": {"outcome": {"name": "Goal"}}, "minute": 10},
            {"match_id": 1, "player": {"name": "A"}, "team": {"name": "Home"}, "type": {"name": "Pass"}, "pass": {}, "minute": 11},
        ]
    )
    match_two = pd.DataFrame(
        [
            {"match_id": 2, "player": {"name": "A"}, "team": {"name": "Home"}, "type": {"name": "Pass"}, "pass": {}, "minute": 11},
        ]
    )
    agg = build_player_aggregates_from_events([match_one, match_two])
    a = agg[agg["player_name"] == "A"].iloc[0]
    assert a["appearances"] == 2
    assert a["total_goals"] == 1
    assert a["total_passes"] == 2
    assert a["total_completed_passes"] == 2


def test_build_player_form_features_validates_columns() -> None:
    with pytest.raises(ValueError):
        build_player_form_features(pd.DataFrame([{"x": 1}]))
