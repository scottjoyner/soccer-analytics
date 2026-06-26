from soccer_edge.models.bundle import save_bundle
from soccer_edge.models.registry import build_registry_index, write_registry_index


def test_build_registry_index(tmp_path) -> None:
    save_bundle({"kind": "demo"}, tmp_path / "demo", "demo", "v1", ["x"], {"accuracy": 1.0})
    index = build_registry_index(tmp_path)
    assert len(index) == 1
    assert index.iloc[0]["name"] == "demo"


def test_write_registry_index(tmp_path) -> None:
    save_bundle({"kind": "demo"}, tmp_path / "demo", "demo", "v1", ["x"], {"accuracy": 1.0})
    path = write_registry_index(tmp_path, tmp_path / "registry.csv")
    assert path.exists()
