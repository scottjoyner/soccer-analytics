import json

from soccer_edge.ingest.openfootball_loader import load_openfootball, parse_score


def test_parse_score_from_openfootball_mapping() -> None:
    assert parse_score({"ft": [2, 1]}) == (2, 1)
    assert parse_score({"home": 0, "away": 0}) == (0, 0)


def test_load_openfootball_json_repo_shape(tmp_path) -> None:
    season_dir = tmp_path / "2024-25"
    season_dir.mkdir()
    (season_dir / "en.1.json").write_text(
        json.dumps(
            {
                "name": "Premier League",
                "season": "2024-25",
                "matches": [
                    {
                        "date": "2024-08-17",
                        "team1": "Arsenal",
                        "team2": "Chelsea",
                        "score": {"ft": [2, 1]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    frame = load_openfootball(tmp_path)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["home_team"] == "Arsenal"
    assert row["away_team"] == "Chelsea"
    assert row["home_score"] == 2
    assert row["result"] == "H"
