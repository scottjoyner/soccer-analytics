from pathlib import Path

from soccer_edge.ingest.metrica_loader import ingest_metrica, load_metrica_events, load_metrica_tracking


def test_metrica_loader_reads_events_and_tracking(tmp_path: Path) -> None:
    (tmp_path / "Sample_Game_1_RawEventsData.csv").write_text("event_id,type\n1,pass\n", encoding="utf-8")
    (tmp_path / "Sample_Game_1_RawTrackingData_Home_Team.csv").write_text("Frame,Time [s]\n1,0.04\n", encoding="utf-8")

    events = load_metrica_events(tmp_path)
    tracking = load_metrica_tracking(tmp_path)
    result = ingest_metrica(tmp_path)

    assert len(events) == 1
    assert len(tracking) == 1
    assert result["status"] == "loaded"
    assert result["events"] == "1"
    assert result["tracking_rows"] == "1"
