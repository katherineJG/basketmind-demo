"""SYNTHETIC batches and graphs for architecture tests only. Never used for reported results."""

import pandas as pd
import torch

from basketmind.graph import GraphSnapshot
from basketmind.models import Batch
from basketmind.utils.config import load_config

N_ITEMS, N_CUSTOMERS, N_COUNTRIES = 40, 10, 4


def model_cfg(**overrides) -> dict:
    cfg = load_config("model")
    cfg["heads"]["reversal_enabled"] = overrides.pop("reversal", False)
    cfg["untie_output"] = overrides.pop("untie", False)
    for name, on in overrides.pop("branches", {}).items():
        cfg["branches"][name] = on
    return cfg


def make_batch(counts=(3, 1, 5), device="cpu", seed=0, max_basket=6) -> Batch:
    """Random batch. Basket sizes run 0..max_basket, so empty baskets occur."""
    g = torch.Generator().manual_seed(seed)
    cfg = load_config("model")
    counts_t = torch.tensor(counts)
    n = int(counts_t.sum())
    sizes = torch.randint(0, max_basket + 1, (n,), generator=g)
    ids = torch.randint(0, N_ITEMS, (int(sizes.sum()),), generator=g)
    offsets = torch.cumsum(sizes, 0) - sizes
    b = len(counts)
    return Batch(
        item_ids=ids,
        item_offsets=offsets,
        basket_counts=counts_t,
        basket_feats=torch.randn(n, cfg["basket_feature_dim"], generator=g),
        customer_idx=torch.randint(0, N_CUSTOMERS + 1, (b,), generator=g),
        country_idx=torch.randint(0, N_COUNTRIES + 1, (b,), generator=g),
        customer_feats=torch.randn(b, cfg["customer_encoder"]["numeric_dim"], generator=g),
    ).to(torch.device(device))


def make_graph(n_nodes=N_ITEMS, n_edges=120, device="cpu", seed=0) -> GraphSnapshot:
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n_nodes, (n_edges,), generator=g)
    dst = torch.randint(0, n_nodes, (n_edges,), generator=g)
    keep = src != dst
    ei = torch.stack([src[keep], dst[keep]])
    return GraphSnapshot(
        ei, torch.rand(ei.shape[1], generator=g) + 0.1, pd.Timestamp("2011-01-01"), n_nodes, 100
    ).to(torch.device(device))
