import numpy as np
import pytest

from soccer_edge.models.cnn_runner import load_tensor_samples, train_cnn_from_npz
from soccer_edge.models.torch_optional import MissingTorchError, torch


def test_load_tensor_samples(tmp_path) -> None:
    path = tmp_path / "samples.npz"
    np.savez(path, spatial=np.zeros((2, 1, 3, 8, 8), dtype=np.float32), labels=np.array([0, 1]))
    samples = load_tensor_samples(path)
    assert len(samples) == 2
    assert samples[0].spatial_sequence.shape == (1, 3, 8, 8)


def test_train_cnn_from_npz_optional_torch(tmp_path) -> None:
    path = tmp_path / "samples.npz"
    np.savez(path, spatial=np.zeros((2, 1, 3, 8, 8), dtype=np.float32), labels=np.array([0, 1]))
    if torch is None:
        with pytest.raises(MissingTorchError):
            train_cnn_from_npz(path, tmp_path / "model", output_classes=2)
        return
    paths = train_cnn_from_npz(path, tmp_path / "model", output_classes=2, epochs=1, batch_size=1, hidden_size=8)
    assert paths["model"].exists()
    assert paths["metadata"].exists()
