"""Best-effort accelerator selection for local training.

AMD Strix Halo (Ryzen AI Max) runs torch under ROCm, which exposes the CUDA device
API, so ``cuda`` is the correct device name there. Apple Silicon exposes ``mps``.
Everything falls back to ``cpu`` when no accelerator is present.
"""

from soccer_edge.models.torch_optional import require_torch, torch


def resolve_device(preferred: str | None = None) -> str:
    """Return the torch device string to use.

    ``preferred`` is honored when it is usable; otherwise pick the best available
    accelerator (cuda/rocm -> mps -> cpu).
    """

    require_torch()
    if preferred is not None:
        try:
            torch.device(preferred)
            return preferred
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
