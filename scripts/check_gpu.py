"""Hardware diagnostic: detection, allocation, forward/backward pass, memory."""

import sys

import torch

from basketmind.utils.device import describe_device, get_device


def main() -> int:
    info = describe_device()
    print(info)
    if info["hip"] is None:
        print("WARNING: not a ROCm build of PyTorch.")
    device = get_device()
    nn = torch.nn
    model = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
    x = torch.randn(32, 128, device=device)
    loss = model(x).pow(2).mean()  # forward
    loss.backward()  # backward
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if device.type == "cuda":
        print(f"allocated MiB: {torch.cuda.memory_allocated() / 2**20:.1f}")
    print(f"OK on {device}: loss={loss.item():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
