"""Write the precomputed-demo bundle that `PrecomputedService` serves.

Runs the REAL service (so it needs torch and the trained model) once for every customer and keeps the model's
top `k` lists under both policies. The bundle holds public data only (the UCI Online Retail II baskets and
products) plus the model's outputs, and `write_bundle` refuses to write anything that looks like a local path.
"""

import json
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from basketmind.inference.data_service import BUNDLE_BASKET_COLUMNS
from basketmind.inference.service import BasketMindService

LOCAL_PATH = re.compile(r"[A-Za-z]:[\\/]|/Users/|/home/|\\\\Users\\\\")


def recommendation_lists(service: BasketMindService, customers: list[str], k: int, log=None) -> dict:
    """Per customer and policy (0 = repeats allowed, 1 = new products only): the model's top `k`."""
    n = len(customers)
    out = {
        "items": np.full((n, 2, k), -1, dtype=np.int16),
        "score": np.zeros((n, 2, k), dtype=np.float32),
        "quantity": np.zeros((n, 2, k), dtype=np.float32),
        "rank": np.zeros((n, 2, k), dtype=np.int16),
    }
    for i, customer in enumerate(customers):
        logits, _ = service.score_customer(customer)
        for policy in (0, 1):
            recs = service.recommend(customer, k, exclude_history=bool(policy))
            idx = np.array([service.item_index[r.product_id] for r in recs], dtype=np.int64)
            m = len(recs)
            out["items"][i, policy, :m] = idx
            out["score"][i, policy, :m] = [r.score for r in recs]
            out["quantity"][i, policy, :m] = [r.predicted_quantity for r in recs]
            if m:  # each product's position among ALL catalogue products, as the explain route reports it
                chosen = logits[torch.as_tensor(idx, device=logits.device)]
                out["rank"][i, policy, :m] = ((logits[None, :] > chosen[:, None]).sum(1) + 1).cpu().numpy()
        if log and (i + 1) % 500 == 0:
            log(f"{i + 1}/{n} customers")
    return out


def write_bundle(service: BasketMindService, out_dir: Path, k: int = 100, log=None) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    customers = sorted(service.by_customer)
    lists = recommendation_lists(service, customers, k, log)

    meta = {
        "items": service.items,
        "popularity": service.popularity.tolist(),
        "train_customers": sorted(service.train_customers),
        "descriptions": service.descriptions,
        "as_of": str(service.as_of),
        "metadata": service.metadata,
        "analytics": service.analytics,
    }
    text = json.dumps(meta, ensure_ascii=False)
    if found := LOCAL_PATH.search(text):
        raise ValueError(f"the bundle would contain a local path ({found.group(0)!r}); refusing to write it")
    (out_dir / "bundle.json").write_text(text, encoding="utf-8")
    service.baskets[BUNDLE_BASKET_COLUMNS].to_parquet(out_dir / "baskets.parquet")
    sp.save_npz(out_dir / "adjacency.npz", service.adjacency.astype(np.float32).tocsr())
    np.savez_compressed(out_dir / "recs.npz", customers=np.array(customers), **lists)
    return {p.name: p.stat().st_size for p in sorted(out_dir.iterdir())}
