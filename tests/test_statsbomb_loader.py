import json
from pathlib import Path

from soccer_edge.ingest.statsbomb_loader import ingest_statsbomb, load_event_files


def test_load_event_files_adds_match_id(tmp_path: Path) -> None:
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    (events_dir / "123.json").write_text(
        json.dumps([
            {
                "id": "event_1",
                "type": {"name": "Pass"},
                "minute": 1,
            }
        ]),
        encoding="utf-8",
    )

    events = load_event_files(tmp_path)
    assert len(events) == 1
    assert events.iloc[0]["match_id"] == "123"


def test_ingest_statsbomb_empty_dir(tmp_path: Path) -> None:
    result = ingest_statsbomb(tmp_path)
    assert result["status"] == "loaded"
    assert result["events"] == "0"
