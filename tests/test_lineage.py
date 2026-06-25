import pandas as pd

from soccer_edge.ingest.lineage import add_lineage_columns, add_lineage_to_tables


def test_add_lineage_columns() -> None:
    frame = pd.DataFrame([{"x": 1}])
    output = add_lineage_columns(frame, source_name="demo", source_path="/tmp/demo", dataset_version="v1")
    assert output.iloc[0]["lineage_source_name"] == "demo"
    assert output.iloc[0]["lineage_dataset_version"] == "v1"
    assert "lineage_ingested_at_utc" in output.columns


def test_add_lineage_to_tables() -> None:
    tables = {"events": pd.DataFrame([{"x": 1}])}
    output = add_lineage_to_tables(tables, source_name="demo", source_path="/tmp/demo")
    assert output["events"].iloc[0]["lineage_source_name"] == "demo"
