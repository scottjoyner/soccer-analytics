from pathlib import Path


from soccer_edge.features.statsbomb_features import (
    build_match_event_features,
    default_event_features,
)


def _write_json(path: Path, obj) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _fixture(root: Path) -> None:
    _write_json(root / "competitions.json", [{"competition_id": 2, "season_id": 27}])
    matches = [
        {
            "match_id": 1,
            "competition": {"competition_id": 2},
            "season": {"season_id": 27},
            "home_team": {"home_team_name": "Home FC"},
            "away_team": {"away_team_name": "Away FC"},
            "home_score": 2,
            "away_score": 1,
        },
        {
            "match_id": 2,
            "competition": {"competition_id": 2},
            "season": {"season_id": 27},
            "home_team": {"home_team_name": "Home FC"},
            "away_team": {"away_team_name": "Away FC"},
            "home_score": 0,
            "away_score": 0,
        },
    ]
    _write_json(root / "matches" / "2" / "27.json", matches)
    events_1 = [
        {"team": {"name": "Home FC"}, "type": {"name": "Shot"}, "shot": {"statsbomb_xg": 0.4, "outcome": {"name": "Goal"}}},
        {"team": {"name": "Home FC"}, "type": {"name": "Pass"}},
        {"team": {"name": "Away FC"}, "type": {"name": "Pressure"}},
    ]
    events_2 = [
        {"team": {"name": "Home FC"}, "type": {"name": "Shot"}, "shot": {"statsbomb_xg": 0.1, "outcome": {"name": "Saved"}}},
        {"team": {"name": "Away FC"}, "type": {"name": "Carry"}},
    ]
    _write_json(root / "events" / "1.json", events_1)
    _write_json(root / "events" / "2.json", events_2)


def test_build_match_event_features(tmp_path: Path) -> None:
    _fixture(tmp_path)
    frame = build_match_event_features(tmp_path)
    assert len(frame) == 2

    row1 = frame[frame["match_id"] == "1"].iloc[0]
    assert row1["home_score"] == 2 and row1["away_score"] == 1
    assert row1["winner"] == 0
    assert row1["home_xg"] == 0.4
    assert row1["home_n_shot"] == 1
    assert row1["home_n_pass"] == 1
    assert row1["away_n_pressure"] == 1

    row2 = frame[frame["match_id"] == "2"].iloc[0]
    assert row2["winner"] == 1
    assert row2["away_n_carry"] == 1
    assert "home_xg" in default_event_features()


def test_build_match_event_features_filters_competition(tmp_path: Path) -> None:
    _fixture(tmp_path)
    frame = build_match_event_features(tmp_path, competition_ids=[999])
    assert len(frame) == 0


def test_default_event_features_unique() -> None:
    cols = default_event_features()
    assert len(cols) == len(set(cols))
    assert "home_xg" in cols and "away_xg" in cols
