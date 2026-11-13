"""Ranking metrics for next-basket recommendation (binary relevance, full-catalogue ranking)."""

import math

import torch


class RankingMetrics:
    """Accumulates Recall@K, Precision@K, NDCG@K and MRR over batches of examples.

    `scores` (B, I) are ranked over the whole catalogue. `targets` (B, I) is bool. With
    `exclude` (B, I) bool (e.g. the customer's history) those items cannot be recommended AND are
    removed from the targets, so the metric measures discovery of new items; examples left with no
    target are skipped and counted in `skipped`. Ties are broken by torch.topk order.
    """

    def __init__(self, ks: list[int]):
        self.ks = sorted(ks)
        self.n = 0
        self.skipped = 0
        names = [f"{m}@{k}" for k in self.ks for m in ("recall", "precision", "ndcg")] + ["mrr"]
        self.sums: dict[str, float] = dict.fromkeys(names, 0.0)  # keys always present (NaN if n == 0)

    def _add(self, name: str, value: torch.Tensor) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(value.sum())

    def update(
        self, scores: torch.Tensor, targets: torch.Tensor, exclude: torch.Tensor | None = None
    ) -> None:
        if exclude is not None:
            scores = scores.masked_fill(exclude, float("-inf"))
            targets = targets & ~exclude
        keep = targets.any(dim=1)
        self.skipped += int((~keep).sum())
        scores, targets = scores[keep], targets[keep]
        if len(scores) == 0:
            return
        n_t = targets.sum(1).float()
        k_max = self.ks[-1]
        top = scores.topk(k_max, dim=1).indices
        hits = targets.gather(1, top).float()  # (B, k_max)
        disc = 1.0 / torch.log2(torch.arange(2, k_max + 2, device=scores.device).float())
        for k in self.ks:
            h = hits[:, :k].sum(1)
            self._add(f"recall@{k}", h / n_t)
            self._add(f"precision@{k}", h / k)
            dcg = (hits[:, :k] * disc[:k]).sum(1)
            ideal = torch.cumsum(disc[:k], 0)[(n_t.clamp(max=k) - 1).long()]
            self._add(f"ndcg@{k}", dcg / ideal)
        best = scores.masked_fill(~targets, float("-inf")).max(1).values
        rank = (scores > best[:, None]).sum(1) + 1
        self._add("mrr", 1.0 / rank.float())
        self.n += len(scores)

    def compute(self) -> dict[str, float]:
        out = {k: v / self.n if self.n else math.nan for k, v in self.sums.items()}
        out["n_evaluated"] = self.n
        out["n_skipped"] = self.skipped
        return out
