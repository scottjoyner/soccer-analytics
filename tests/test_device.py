import torch

from soccer_edge.models.device import resolve_device


def test_resolve_device_preferred_is_honored() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_is_valid_torch_device() -> None:
    device = resolve_device()
    assert torch.device(device) is not None


def test_resolve_device_invalid_preferred_falls_back() -> None:
    device = resolve_device("not-a-real-device")
    assert device in {"cuda", "mps", "cpu"}
