from pathlib import Path

import numpy as np


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
        {"team": {"name": "Home FC"}, "type": {"name": "Pass"}, "location": [60, 40], "pass": {"end_location": [70, 40]}},
        {"team": {"name": "Home FC"}, "type": {"name": "Pressure"}, "location": [50, 40]},
        {"team": {"name": "Home FC"}, "type": {"name": "Ball Recovery"}, "location": [52, 41]},
        {"team": {"name": "Away FC"}, "type": {"name": "Pressure"}, "location": [30, 30]},
        {"team": {"name": "Home FC"}, "type": {"name": "Shot"}, "location": [110, 40],
         "shot": {"statsbomb_xg": 0.4, "outcome": {"name": "Goal"}, "end_location": [120, 40]}},
    ]
    events_2 = [
        {"team": {"name": "Home FC"}, "type": {"name": "Shot"}, "location": [108, 40],
         "shot": {"statsbomb_xg": 0.1, "outcome": {"name": "Saved"}, "end_location": [120, 40]}},
        {"team": {"name": "Away FC"}, "type": {"name": "Carry"}, "location": [40, 30], "carry": {"end_location": [45, 30]}},
        {"team": {"name": "Away FC"}, "type": {"name": "Pressure"}, "location": [45, 30]},
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
    # xT and pressure regains are produced and the press was regained once.
    assert "home_xt" in frame.columns and "home_pressure_regains" in frame.columns
    assert row1["home_pressure_regains"] == 1
    assert row1["home_xt"] > 0.0

    row2 = frame[frame["match_id"] == "2"].iloc[0]
    assert row2["winner"] == 1
    assert row2["away_n_carry"] == 1
    assert "home_xg" in default_event_features()


def test_build_match_event_features_filters_competition(tmp_path: Path) -> None:
    _fixture(tmp_path)
    frame = build_match_event_features(tmp_path, competition_ids=[999])
    assert len(frame) == 0


def test_default_event_features_includes_xt_and_pressure(tmp_path: Path) -> None:
    _fixture(tmp_path)
    build_match_event_features(tmp_path)  # ensure module imports/works
    cols = default_event_features()
    assert len(cols) == len(set(cols))
    assert "home_xg" in cols and "away_xg" in cols
    assert "home_xt" in cols and "away_pressure_regains" in cols


def test_per_fold_xt_differs_from_league(tmp_path: Path) -> None:
    from soccer_edge.features.statsbomb_features import (
        build_match_event_features,
        build_match_event_features_fold,
    )

    _fixture(tmp_path)
    league = build_match_event_features(tmp_path)
    fold = build_match_event_features_fold(tmp_path, train_match_ids=["1"])

    # Match 2's xT uses a surface fit only on match 1, so it differs from the
    # league-wide surface that also saw match 2.
    lg2 = league[league["match_id"] == "2"].iloc[0]
    fd2 = fold[fold["match_id"] == "2"].iloc[0]
    assert lg2["home_xt"] != fd2["home_xt"]


def test_per_fold_xt_full_matches_league(tmp_path: Path) -> None:
    from soccer_edge.features.statsbomb_features import (
        build_match_event_features,
        build_match_event_features_fold,
    )

    _fixture(tmp_path)
    league = build_match_event_features(tmp_path).set_index("match_id").sort_index()
    fold = build_match_event_features_fold(tmp_path, train_match_ids=["1", "2"]).set_index("match_id").sort_index()
    for col in ("home_xt", "away_xt"):
        assert np.allclose(league[col].to_numpy(), fold[col].to_numpy())
