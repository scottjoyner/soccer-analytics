from pathlib import Path

from soccer_edge.example_pipeline import run_tiny_example_pipeline


def test_run_tiny_pipeline(tmp_path) -> None:
    paths = run_tiny_example_pipeline(Path("."), tmp_path / "out")
    assert paths["predictions"].exists()
    assert paths["markdown"].exists()
    assert paths["tensor_samples"].exists()
