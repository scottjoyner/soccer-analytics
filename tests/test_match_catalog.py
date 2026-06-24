from pathlib import Path

from soccer_edge.ingest.match_catalog import read_match_catalog


def test_read_match_catalog(tmp_path: Path) -> None:
    catalog = tmp_path / "matches.csv"
    catalog.write_text(
        "match_id,competition,season,match_date,stage,home_team,away_team,venue,home_score,away_score,status\n"
        "match_1,FIFA World Cup,example,2026-06-11,group,A,B,Stadium,,,scheduled\n",
        encoding="utf-8",
    )

    rows = read_match_catalog(catalog)
    assert len(rows) == 1
    assert rows[0].match_id == "match_1"
    assert rows[0].home_team == "A"
