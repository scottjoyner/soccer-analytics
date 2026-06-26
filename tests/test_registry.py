from soccer_edge.models.bundle import save_bundle
from soccer_edge.models.registry import build_registry_index, summarize_registry, write_registry_index, write_registry_summary


def test_build_registry_index(tmp_path) -> None:
    save_bundle({"kind": "demo"}, tmp_path / "demo", "demo", "v1", ["x"], {"accuracy": 1.0})
    index = build_registry_index(tmp_path)
    assert len(index) == 1
    assert index.iloc[0]["name"] == "demo"
    assert index.iloc[0]["metric_accuracy"] == 1.0


def test_summarize_registry() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"name": "a", "metric_accuracy": 0.5, "created_at_utc": "2026-01-01T00:00:00+00:00"},
            {"name": "b", "metric_accuracy": 0.9, "created_at_utc": "2026-01-02T00:00:00+00:00"},
        ]
    )
    summary = summarize_registry(frame)
    assert summary.iloc[0]["name"] == "b"


def test_write_registry_index(tmp_path) -> None:
    save_bundle({"kind": "demo"}, tmp_path / "demo", "demo", "v1", ["x"], {"accuracy": 1.0})
    path = write_registry_index(tmp_path, tmp_path / "registry.csv")
    assert path.exists()


def test_write_registry_summary(tmp_path) -> None:
    save_bundle({"kind": "demo"}, tmp_path / "demo", "demo", "v1", ["x"], {"accuracy": 1.0})
    path = write_registry_summary(tmp_path, tmp_path / "summary.csv")
    assert path.exists()
