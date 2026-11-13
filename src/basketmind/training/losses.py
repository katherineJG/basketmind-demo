"""Multi-task loss: L_total = w_basket * L_basket + w_quantity * L_quantity (+ w_reversal * L_reversal)."""

from typing import Any

import torch
import torch.nn.functional as F

from basketmind.training.data import Targets


def multitask_loss(
    out: dict[str, torch.Tensor],
    targets: Targets,
    loss_weights: dict[str, float],
    pos_weight: float,
    huber_delta: float,
) -> dict[str, torch.Tensor]:
    """Weighted BCE over the whole catalogue plus Huber on log1p quantity of sampled target items.

    The reversal term is omitted: it needs validated cancellation labels, which do not exist yet, so a
    non-zero reversal weight is refused instead of silently ignored.
    """
    if loss_weights.get("reversal", 0.0) > 0 or "reversal" in out:
        raise ValueError("reversal loss is not available until cancellation labels are validated")
    logits = out["basket_logits"]
    basket = F.binary_cross_entropy_with_logits(
        logits, targets.basket_hot.to(logits.dtype), pos_weight=torch.tensor(pos_weight, device=logits.device)
    )
    mask = targets.quantity_mask
    quantity = (
        F.huber_loss(out["quantity"][mask], targets.quantity_target[mask], delta=huber_delta)
        if mask.any()
        else logits.new_zeros(())
    )
    total = loss_weights["basket"] * basket + loss_weights["quantity"] * quantity
    return {"total": total, "basket": basket, "quantity": quantity}


def loss_settings(train_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss_weights": model_cfg["loss_weights"],
        "pos_weight": train_cfg["pos_weight"],
        "huber_delta": train_cfg["huber_delta"],
    }
