"""Device detection: AMD GPU via ROCm when available, otherwise CPU."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Resolve 'auto' | 'cpu' | 'cuda'. On ROCm builds the torch.cuda API is the AMD GPU."""
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "cuda":
        raise RuntimeError("device='cuda' requested but no GPU is available")
    return torch.device("cpu")


def describe_device() -> dict[str, object]:
    """Facts about the active backend, for logs and experiment metadata."""
    on_gpu = torch.cuda.is_available()
    return {
        "torch": torch.__version__,
        "hip": torch.version.hip,  # None on non-ROCm builds
        "gpu_available": on_gpu,
        "gpu_name": torch.cuda.get_device_name(0) if on_gpu else None,
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        if on_gpu
        else None,
    }
