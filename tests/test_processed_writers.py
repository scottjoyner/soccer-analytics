import pandas as pd

from soccer_edge.ingest.processed_tables import write_metrica_processed, write_processed_table


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
