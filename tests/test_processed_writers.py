import pandas as pd

from soccer_edge.ingest.processed_tables import write_metrica_processed, write_processed_table, write_soccernet_processed


def test_write_processed_table(tmp_path) -> None:
    path = write_processed_table(pd.DataFrame([{"x": 1}]), tmp_path, "demo")
    assert path.exists()


def test_write_metrica_processed(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "RawEvents.csv").write_text("event_id,type\n1,pass\n", encoding="utf-8")
    paths = write_metrica_processed(source, output, dataset_version="v1")
    assert paths["metrica_events"].exists()


def test_write_soccernet_processed(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "actions.json").write_text('[{"action_id": 1, "type": "pass"}]', encoding="utf-8")
    (source / "labels.csv").write_text("frame,label\n1,player\n", encoding="utf-8")
    paths = write_soccernet_processed(source, output, dataset_version="v2")
    assert paths["soccernet_json"].exists()
    assert paths["soccernet_csv"].exists()
