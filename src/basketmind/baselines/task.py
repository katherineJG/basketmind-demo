"""Baseline task: train-only catalogue, vocabulary-mapped baskets and evaluation examples."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


@dataclass
class Task:
    items: list[str]  # catalogue: products seen in TRAIN baskets, index = position
    basket_items: dict[str, np.ndarray]  # invoice -> sorted unique catalogue indices
    examples: dict[str, pd.DataFrame]  # split -> examples (with `target_idx`), non-empty targets only
    train_matrix: sp.csr_matrix  # train customers x items, binary
    train_customers: dict[str, int]  # customer id -> row in train_matrix
    popularity: np.ndarray  # train baskets containing each item
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def history_hot(self, batch: pd.DataFrame, device: torch.device) -> torch.Tensor:
        """Binary (B, n_items) matrix of items in each example's history baskets."""
        out = torch.zeros(len(batch), self.n_items, device=device)
        for row, hist in enumerate(batch["history_invoices"]):
            idx = [self.basket_items[i] for i in hist]
            if idx:
                out[row, torch.as_tensor(np.concatenate(idx), device=device)] = 1.0
        return out

    def history_weighted(self, batch: pd.DataFrame, device: torch.device, decay: float = 1.0) -> torch.Tensor:
        """(B, n_items) purchase counts from each example's history, oldest basket first.

        A basket `n` baskets older than the newest one counts `decay ** n`, so decay=1 is a plain count and
        decay<1 favours recent purchases. History invoices are in chronological order.
        """
        out = np.zeros((len(batch), self.n_items), dtype=np.float32)
        for row, hist in enumerate(batch["history_invoices"]):
            for age, inv in enumerate(reversed(list(hist))):
                out[row, self.basket_items[inv]] += decay**age
        return torch.as_tensor(out, device=device)

    def target_hot(self, batch: pd.DataFrame, device: torch.device) -> torch.Tensor:
        out = torch.zeros(len(batch), self.n_items, dtype=torch.bool, device=device)
        for row, idx in enumerate(batch["target_idx"]):
            out[row, torch.as_tensor(idx, device=device)] = True
        return out


def load_task(processed_dir: str | Path) -> Task:
    """Build the task from the processed baskets and examples. The catalogue comes from TRAIN only."""
    d = Path(processed_dir)
    baskets = pd.read_parquet(d / "baskets.parquet")
    examples = pd.read_parquet(d / "next_basket_examples.parquet")

    train = baskets[baskets["split"] == "train"]
    items = sorted(train["items"].explode().unique())
    vocab = {s: i for i, s in enumerate(items)}

    basket_items = {
        inv: np.array(sorted({vocab[s] for s in its if s in vocab}), dtype=np.int64)
        for inv, its in zip(baskets["InvoiceNo"], baskets["items"], strict=True)
    }

    cust_ids = sorted(train["CustomerID"].unique())
    train_customers = {c: i for i, c in enumerate(cust_ids)}
    rows, cols = [], []
    for c, inv in zip(train["CustomerID"], train["InvoiceNo"], strict=True):
        idx = basket_items[inv]
        rows.extend([train_customers[c]] * len(idx))
        cols.extend(idx.tolist())
    mat = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(cust_ids), len(items))
    )
    mat.data[:] = 1.0  # binary even where a customer bought an item in several baskets

    pop = np.zeros(len(items), dtype=np.float64)
    for inv in train["InvoiceNo"]:
        pop[basket_items[inv]] += 1

    examples = examples.copy()
    examples["target_idx"] = examples["target_invoice"].map(basket_items)
    stats: dict[str, Any] = {"catalogue_size": len(items)}
    per_split = {}
    for split in ("train", "validation", "test"):
        e = examples[examples["split"] == split]
        raw = baskets.set_index("InvoiceNo")["n_items"].reindex(e["target_invoice"]).to_numpy()
        kept = e["target_idx"].map(len)
        per_split[split] = {
            "examples": len(e),
            "examples_with_in_catalogue_target": int((kept > 0).sum()),
            "target_items_total": int(raw.sum()),
            "target_items_in_catalogue": int(kept.sum()),
        }
    stats["splits"] = per_split
    ex = {
        s: examples[(examples["split"] == s) & (examples["target_idx"].map(len) > 0)].reset_index(drop=True)
        for s in ("train", "validation", "test")
    }
    return Task(items, basket_items, ex, mat, train_customers, pop, stats)
