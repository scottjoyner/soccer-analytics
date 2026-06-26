import numpy as np
import pytest

from soccer_edge.models.cnn_predict import export_cnn_bundle_predictions
from soccer_edge.models.cnn_runner import train_cnn_from_npz
from soccer_edge.models.torch_optional import torch


def test_cnn_output_writer(tmp_path) -> None:
    if torch is None:
        pytest.skip("torch is optional")
    source = tmp_path / "samples.npz"
    np.savez(source, spatial=np.zeros((2, 1, 3, 8, 8), dtype=np.float32), labels=np.array([0, 1]))
    bundle = tmp_path / "bundle"
    train_cnn_from_npz(source, bundle, output_classes=2, epochs=1, batch_size=1, hidden_size=8)
    output = export_cnn_bundle_predictions(bundle, source, tmp_path / "out.csv", batch_size=1)
    assert output.exists()
