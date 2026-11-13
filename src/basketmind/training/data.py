"""Training data: examples grouped by graph snapshot, collated into model batches and targets."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from basketmind.baselines.task import Task, load_task
from basketmind.features.build import FeatureStore, build_feature_store
from basketmind.graph import GraphSnapshot, create_graph_snapshots
from basketmind.models.batch import Batch
from basketmind.utils.logging import get_logger

log = get_logger("training.data")


@dataclass
class Targets:
    basket_hot: torch.Tensor  # (B, n_items) bool
    quantity_items: torch.Tensor  # (B, K) long (0 where masked)
    quantity_target: torch.Tensor  # (B, K) float: log1p(quantity)
    quantity_mask: torch.Tensor  # (B, K) bool
    history_hot: torch.Tensor  # (B, n_items) bool: items in the history (for the exclude_history policy)

    def to(self, device: torch.device) -> "Targets":
        return Targets(**{k: v.to(device) for k, v in self.__dict__.items()})


@dataclass
class TrainingData:
    task: Task
    store: FeatureStore
    boundaries: list[pd.Timestamp]
    snapshots: dict[pd.Timestamp, GraphSnapshot]  # monthly, kept on the CPU
    window: dict[str, np.ndarray]  # split -> snapshot index of every example
    quantity_samples: int

    def snapshot_for(self, window_idx: int) -> GraphSnapshot:
        return self.snapshots[self.boundaries[window_idx]]

    def n_examples(self, split: str) -> int:
        return len(self.task.examples[split])

    def iter_batches(
        self, split: str, batch_size: int, rng: np.random.Generator | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """(window index, example indices). Every batch lies inside one snapshot window. Shuffles if `rng`."""
        win = self.window[split]
        batches = []
        for w in np.unique(win):
            idx = np.flatnonzero(win == w)
            if rng is not None:
                idx = rng.permutation(idx)
            batches += [(int(w), idx[s : s + batch_size]) for s in range(0, len(idx), batch_size)]
        if rng is not None:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        yield from batches

    def collate(
        self, split: str, idx: np.ndarray, rng: np.random.Generator | None = None
    ) -> tuple[Batch, Targets]:
        """Inputs and targets for examples `idx`; `rng` picks which target items train the quantity head."""
        task, store, ex = self.task, self.store, self.task.examples[split]
        arr = store.example_arrays[split]
        hist = [ex["history_invoices"].iat[i] for i in idx]
        flat = [inv for h in hist for inv in h]
        item_arrays = [task.basket_items[inv] for inv in flat]
        sizes = np.array([len(a) for a in item_arrays])
        batch = Batch(
            item_ids=torch.as_tensor(
                np.concatenate(item_arrays) if sizes.sum() else np.zeros(0, dtype=np.int64)
            ),
            item_offsets=torch.as_tensor(np.cumsum(sizes) - sizes),
            basket_counts=torch.as_tensor([len(h) for h in hist]),
            basket_feats=torch.as_tensor(np.stack([store.basket_feats[inv] for inv in flat])),
            customer_idx=torch.as_tensor(arr["customer_idx"][idx]),
            country_idx=torch.as_tensor(arr["country_idx"][idx]),
            customer_feats=torch.as_tensor(arr["customer_feats"][idx]),
        )
        b, n_items, k = len(idx), task.n_items, self.quantity_samples
        basket_hot = torch.zeros(b, n_items, dtype=torch.bool)
        history_hot = torch.zeros(b, n_items, dtype=torch.bool)
        q_items = torch.zeros(b, k, dtype=torch.long)
        q_target = torch.zeros(b, k)
        q_mask = torch.zeros(b, k, dtype=torch.bool)
        for row, i in enumerate(idx):
            basket_hot[row, torch.as_tensor(ex["target_idx"].iat[i])] = True
            past = [task.basket_items[inv] for inv in ex["history_invoices"].iat[i]]
            history_hot[row, torch.as_tensor(np.concatenate(past))] = True
            items, qty = store.quantities[ex["target_invoice"].iat[i]]
            pick = rng.permutation(len(items))[:k] if rng is not None else np.arange(min(k, len(items)))
            q_items[row, : len(pick)] = torch.as_tensor(items[pick])
            q_target[row, : len(pick)] = torch.as_tensor(np.log1p(np.maximum(qty[pick], 0)))
            q_mask[row, : len(pick)] = True
        return batch, Targets(basket_hot, q_items, q_target, q_mask, history_hot)


def monthly_boundaries(baskets: pd.DataFrame, every_months: int) -> list[pd.Timestamp]:
    start = baskets["timestamp"].min().to_period("M").to_timestamp()
    end = baskets["timestamp"].max()
    return list(pd.date_range(start, end, freq=pd.DateOffset(months=every_months)))


def build_training_data(
    processed_dir: str | Path, feat_cfg: dict[str, Any], graph_cfg: dict[str, Any], quantity_samples: int
) -> TrainingData:
    """Load the processed data and build everything the training loop needs (features, snapshots, windows)."""
    processed_dir = Path(processed_dir)
    task = load_task(processed_dir)
    baskets = pd.read_parquet(processed_dir / "baskets.parquet")
    store = build_feature_store(task, baskets, feat_cfg)
    bounds = monthly_boundaries(baskets, feat_cfg["snapshot_frequency_months"])
    snaps = create_graph_snapshots(baskets, bounds, task.items, graph_cfg)
    log.info(
        "%d snapshots, edges per snapshot: %s", len(snaps), [s.edge_index.shape[1] for s in snaps.values()]
    )
    stamps = np.array(bounds, dtype="datetime64[ns]")
    window = {
        s: np.searchsorted(stamps, ex["target_timestamp"].to_numpy(dtype="datetime64[ns]"), side="right") - 1
        for s, ex in task.examples.items()
    }
    return TrainingData(task, store, bounds, snaps, window, quantity_samples)
