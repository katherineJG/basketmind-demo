"""Diagnostic (validation only): does a trained model's score depend on the customer's history?

Scores validation four ways. In the swapped variants the history and/or the customer information of each
example is replaced by that of another example from the same batch (same graph snapshot, same month), while
the targets stay put. If swapping barely changes the score, the model is not using that input.
"""

import argparse

import numpy as np
import torch

from basketmind.evaluation.ranking import RankingMetrics
from basketmind.models import Batch
from basketmind.training.checkpoint import load_checkpoint
from basketmind.training.data import build_training_data
from basketmind.utils.config import load_config, project_root
from basketmind.utils.device import get_device, release_gpu

HISTORY = ("item_ids", "item_offsets", "basket_counts", "basket_feats")
CUSTOMER = ("customer_idx", "country_idx", "customer_feats")
VARIANTS = {
    "normal": (False, False),
    "histories swapped": (True, False),
    "customer info swapped": (False, True),
    "both swapped": (True, True),
}


def mix(own: Batch, other: Batch, swap_history: bool, swap_customer: bool) -> Batch:
    fields = dict(own.__dict__)
    for names, swap in ((HISTORY, swap_history), (CUSTOMER, swap_customer)):
        if swap:
            fields |= {n: getattr(other, n) for n in names}
    return Batch(**fields)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="experiments/neural/full/seed42/best.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = project_root()
    train_cfg = load_config("training")
    device = get_device(train_cfg["device"])
    model, payload = load_checkpoint(root / args.checkpoint, device)
    model.eval()
    data = build_training_data(
        root / "data" / "processed",
        load_config("features"),
        load_config("graph"),
        train_cfg["quantity_samples"],
    )
    meta = payload["metadata"]
    print(f"checkpoint: epoch {meta['epoch']}, validation score {meta['val_score']:.4f}")
    for name, (swap_h, swap_c) in VARIANTS.items():
        rng = np.random.default_rng(args.seed)
        meter = RankingMetrics([5, 10, 20])
        for window, idx in data.iter_batches("validation", train_cfg["batch_size"]):
            own, targets = data.collate("validation", idx)
            other, _ = data.collate("validation", idx[rng.permutation(len(idx))])
            batch = mix(own, other, swap_h, swap_c).to(device)
            out = model(batch, data.snapshot_for(window).to(device))
            meter.update(out["basket_logits"], targets.basket_hot.to(device))
        r = meter.compute()
        print(
            f"{name:24s} recall@10={r['recall@10']:.4f} ndcg@10={r['ndcg@10']:.4f} mrr={r['mrr']:.4f}",
            flush=True,
        )
    release_gpu()


if __name__ == "__main__":
    main()
