import math

import pytest
import torch

from basketmind.training.data import Targets
from basketmind.training.losses import multitask_loss

W = {"basket": 2.0, "quantity": 0.5, "reversal": 0.0}


def targets(hot, q_target, q_mask):
    b, k = len(hot), len(q_target[0])
    return Targets(
        torch.tensor(hot),
        torch.zeros(b, k, dtype=torch.long),
        torch.tensor(q_target),
        torch.tensor(q_mask),
        torch.zeros(b, len(hot[0]), dtype=torch.bool),
    )


def test_weighted_bce_and_huber_hand_computed():
    out = {"basket_logits": torch.zeros(1, 2), "quantity": torch.tensor([[1.5, 9.9]])}
    t = targets([[True, False]], [[1.0, 0.0]], [[True, False]])  # second quantity slot is masked out
    losses = multitask_loss(out, t, W, pos_weight=3.0, huber_delta=1.0)
    # logits 0: p = 0.5. positive: 3 * ln 2, negative: ln 2, mean over both entries = 2 ln 2
    assert losses["basket"].item() == pytest.approx(2 * math.log(2))
    assert losses["quantity"].item() == pytest.approx(0.5 * 0.5**2)  # |1.5 - 1.0| <= delta: 0.5 e^2
    assert losses["total"].item() == pytest.approx(2.0 * 2 * math.log(2) + 0.5 * 0.125)


def test_huber_is_linear_beyond_delta():
    out = {"basket_logits": torch.zeros(1, 2), "quantity": torch.tensor([[4.0]])}
    t = targets([[True, False]], [[1.0]], [[True]])
    q = multitask_loss(out, t, W, 1.0, huber_delta=1.0)["quantity"].item()
    assert q == pytest.approx(1.0 * (3.0 - 0.5))  # delta * (|e| - delta / 2)


def test_no_quantity_targets_gives_zero_quantity_loss():
    out = {"basket_logits": torch.zeros(1, 2), "quantity": torch.tensor([[1.0]])}
    t = targets([[True, False]], [[0.0]], [[False]])
    losses = multitask_loss(out, t, W, 1.0, 1.0)
    assert losses["quantity"].item() == 0.0 and torch.isfinite(losses["total"])


def test_reversal_loss_is_refused_not_ignored():
    out = {"basket_logits": torch.zeros(1, 2), "quantity": torch.zeros(1, 1)}
    t = targets([[True, False]], [[0.0]], [[True]])
    with pytest.raises(ValueError, match="cancellation labels"):
        multitask_loss(out, t, {**W, "reversal": 1.0}, 1.0, 1.0)
    with pytest.raises(ValueError, match="cancellation labels"):
        multitask_loss({**out, "reversal": torch.zeros(1, 1)}, t, W, 1.0, 1.0)


def test_loss_is_differentiable_through_both_tasks():
    logits = torch.zeros(1, 2, requires_grad=True)
    quantity = torch.zeros(1, 1, requires_grad=True)
    t = targets([[True, False]], [[1.0]], [[True]])
    multitask_loss({"basket_logits": logits, "quantity": quantity}, t, W, 2.0, 1.0)["total"].backward()
    assert logits.grad.abs().sum() > 0 and quantity.grad.abs().sum() > 0
