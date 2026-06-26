from pathlib import Path

from soccer_edge.ingest.processed_tables import write_statsbomb_processed


def test_write_statsbomb_processed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "matches" / "1").mkdir(parents=True)
    (source / "events").mkdir(parents=True)
    (source / "lineups").mkdir(parents=True)
    (source / "competitions.json").write_text('[{"competition_id": 1, "season_id": 1}]', encoding="utf-8")
    (source / "matches" / "1" / "1.json").write_text('[{"match_id": 1, "home_team": {"home_team_name": "A"}}]', encoding="utf-8")
    (source / "events" / "1.json").write_text('[{"id": "e1", "type": {"name": "Pass"}}]', encoding="utf-8")
    (source / "lineups" / "1.json").write_text('[{"team_name": "A", "lineup": []}]', encoding="utf-8")

    paths = write_statsbomb_processed(source, output, dataset_version="v1")
    assert paths["statsbomb_competitions"].exists()
    assert paths["statsbomb_matches"].exists()
    assert paths["statsbomb_events"].exists()
    assert paths["statsbomb_lineups"].exists()
