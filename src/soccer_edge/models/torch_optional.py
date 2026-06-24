try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


class MissingTorchError(RuntimeError):
    pass


def require_torch() -> None:
    if torch is None or nn is None:
        raise MissingTorchError("Install optional ML dependencies with: pip install -r requirements-ml.txt")
