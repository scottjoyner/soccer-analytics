import pandas as pd

from soccer_edge.raw_data_sources import default_raw_data_sources, raw_data_sources_frame, write_raw_data_sources


def test_default_raw_data_sources() -> None:
    sources = default_raw_data_sources()
    assert any(source.name == "StatsBomb Open Data" for source in sources)
    assert any("player" in source.player_stats_level for source in sources)


def test_raw_data_sources_frame() -> None:
    frame = raw_data_sources_frame()
    assert "rights_posture" in frame.columns
    assert "player_stats_level" in frame.columns


def test_write_raw_data_sources(tmp_path) -> None:
    output = write_raw_data_sources(tmp_path / "sources.csv")
    assert output.exists()
    assert len(pd.read_csv(output)) >= 5
