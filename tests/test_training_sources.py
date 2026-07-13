import pandas as pd

from soccer_edge.training_sources import default_training_sources, training_sources_frame, write_training_sources


def test_default_training_sources() -> None:
    sources = default_training_sources()
    assert any(source.name == "StatsBomb Open Data" for source in sources)
    assert any(source.tier == "local-rights-approved" for source in sources)


def test_training_sources_frame() -> None:
    frame = training_sources_frame()
    assert "rights_posture" in frame.columns
    assert len(frame) >= 4


def test_write_training_sources(tmp_path) -> None:
    output = write_training_sources(tmp_path / "sources.csv")
    assert output.exists()
    assert len(pd.read_csv(output)) >= 4
