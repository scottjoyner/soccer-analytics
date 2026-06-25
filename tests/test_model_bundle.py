from soccer_edge.models.bundle import load_bundle, save_bundle
from soccer_edge.models.run_metadata import RunMetadata, read_run_metadata, write_run_metadata


def test_run_metadata_roundtrip(tmp_path):
    path = tmp_path / "metadata.json"
    metadata = RunMetadata(
        name="demo",
        version="v1",
        created_at_utc="2026-01-01T00:00:00+00:00",
        feature_names=["x"],
        metrics={"accuracy": 1.0},
    )
    write_run_metadata(metadata, path)
    assert read_run_metadata(path) == metadata


def test_save_and_load_bundle(tmp_path):
    model = {"kind": "demo"}
    paths = save_bundle(
        model=model,
        output_dir=tmp_path,
        name="demo",
        version="v1",
        feature_names=["x"],
        metrics={"accuracy": 1.0},
    )
    assert paths["model"].exists()
    assert paths["metadata"].exists()
    loaded_model, metadata = load_bundle(tmp_path)
    assert loaded_model == model
    assert metadata.name == "demo"
