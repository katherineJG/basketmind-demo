"""Regression tests for ops that are known to fail on ROCm for Windows. Runs on the default device."""

import numpy as np
import torch

from basketmind.baselines.models import _GRUNet
from basketmind.utils.device import get_device


def test_gru_net_backward_works_on_default_device():
    dev = get_device()
    net = _GRUNet(n_items=30, item_dim=8, hidden=16, dropout=0.0).to(dev)
    hist = [
        [np.array([1, 2, 3]), np.array([4]), np.array([5, 6])],  # 3 baskets
        [np.array([7, 8])],  # 1 basket, exercises the padding mask
    ]
    out = net(hist, dev)
    assert out.shape == (2, 30) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_padding_does_not_change_a_short_histories_output():
    net = _GRUNet(n_items=30, item_dim=8, hidden=16, dropout=0.0).eval()
    short = [[np.array([7, 8])]]
    alone = net(short, torch.device("cpu"))
    padded = net(short + [[np.array([1]), np.array([2]), np.array([3])]], torch.device("cpu"))[:1]
    assert torch.allclose(alone, padded, atol=1e-6)
