"""Checkpoints: weights, constructor arguments, config and metadata in one file."""

from pathlib import Path
from typing import Any

import torch

from basketmind.models import BasketMindModel


def save_checkpoint(
    path: str | Path,
    model: BasketMindModel,
    metadata: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a checkpoint. Refuses to replace an existing file unless `overwrite` (never lose a best model).

    `metadata` must hold plain Python values (str, numbers, lists, dicts) because loading is weights-only.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True or archive it first")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_args": model.init_args,
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "metadata": metadata or {},
            "torch_version": str(
                torch.__version__
            ),  # plain str: TorchVersion is rejected by weights-only load
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[BasketMindModel, dict[str, Any]]:
    """Rebuild the model from the stored arguments and load its weights. Returns (model, payload)."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    model = BasketMindModel(**payload["model_args"]).to(device)
    model.load_state_dict(payload["model_state"])
    return model, payload
