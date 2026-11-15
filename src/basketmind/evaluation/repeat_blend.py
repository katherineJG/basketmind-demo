"""Give BasketMind a repeat-purchase score: its logits plus a weighted personal-frequency term.

score = logit + lam * count / max(count), where count is the customer's recency-weighted purchase count of
each item (Task.history_weighted). `lam` and the recency `decay` are chosen on VALIDATION Recall@10 (policy
`full`); lam = 0 is the unchanged model, so the blend can only match or beat it on validation. Nothing is
retrained. Items the customer never bought get no bonus, so under `exclude_history` the ranking is identical
to the unchanged model (the caller checks this against the reported numbers).
"""

from dataclasses import dataclass

import numpy as np
import torch

from basketmind.evaluation.ranking import RankingMetrics
from basketmind.training.data import TrainingData

LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
DECAYS = (1.0, 0.9, 0.8)
POLICIES = ("full", "exclude_history")


@dataclass
class Cache:
    """One split's model logits, history counts (per decay), targets and history masks, kept on the device."""

    logits: list[torch.Tensor]
    counts: dict[float, list[torch.Tensor]]
    targets: list[torch.Tensor]
    history: list[torch.Tensor]
    ids: list[np.ndarray]


@torch.no_grad()
def build_cache(
    net: torch.nn.Module,
    data: TrainingData,
    split: str,
    decays: tuple[float, ...],
    device: torch.device,
    batch_size: int = 128,
) -> Cache:
    net.eval()
    ex = data.task.examples[split]
    cache = Cache([], {d: [] for d in decays}, [], [], [])
    for window, idx in data.iter_batches(split, batch_size):
        batch, targets = data.collate(split, idx)
        cache.logits.append(net(batch.to(device), data.snapshot_for(window).to(device))["basket_logits"])
        rows = ex.iloc[idx]
        for d in decays:
            cache.counts[d].append(data.task.history_weighted(rows, device, d))
        cache.targets.append(targets.basket_hot.to(device))
        cache.history.append(targets.history_hot.to(device))
        cache.ids.append(idx)
    return cache


def blend(logits: torch.Tensor, counts: torch.Tensor, lam: float) -> torch.Tensor:
    return logits + lam * counts / counts.amax(1, keepdim=True).clamp(min=1e-9)


def evaluate(
    cache: Cache, decay: float, lam: float, ks: list[int], policy: str, keep: bool = False
) -> RankingMetrics:
    meter = RankingMetrics(ks, keep_examples=keep)
    for i, logits in enumerate(cache.logits):
        scores = blend(logits, cache.counts[decay][i], lam)
        exclude = cache.history[i] if policy == "exclude_history" else None
        meter.update(scores, cache.targets[i], exclude, ids=cache.ids[i])
    return meter


def choose(
    cache: Cache, ks: list[int], decays=DECAYS, lambdas=LAMBDAS, metric: str = "recall@10"
) -> tuple[tuple[float, float], list[dict]]:
    """Best (decay, lam) on this (validation) cache by `metric` under policy `full`, and the whole grid."""
    grid = [
        {"decay": d, "lam": lam, "val": evaluate(cache, d, lam, ks, "full").compute()[metric]}
        for d in decays
        for lam in lambdas
    ]
    best = max(grid, key=lambda r: r["val"])  # ties keep the first cell in grid order
    return (best["decay"], best["lam"]), grid
