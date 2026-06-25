from pathlib import Path

from soccer_edge.ingest.soccernet_loader import ingest_soccernet, load_soccernet_csv_files, load_soccernet_json_files


def test_soccernet_loader_reads_json_and_csv(tmp_path: Path) -> None:
    (tmp_path / "labels.json").write_text('[{"gameTime": "1 - 00:01", "label": "pass"}]', encoding="utf-8")
    (tmp_path / "tracking.csv").write_text("frame,x,y\n1,0.1,0.2\n", encoding="utf-8")

    json_rows = load_soccernet_json_files(tmp_path)
    csv_rows = load_soccernet_csv_files(tmp_path)
    result = ingest_soccernet(tmp_path)

    assert len(json_rows) == 1
    assert len(csv_rows) == 1
    assert result["status"] == "loaded"
    assert result["json_rows"] == "1"
    assert result["csv_rows"] == "1"
