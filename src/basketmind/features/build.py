"""Basket and customer features. Everything an example uses was observable before its target basket.

Normalisers (mean/std) are fitted on TRAIN baskets / TRAIN examples only and then applied to every split.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from basketmind.baselines.task import Task

BASKET_LOG_COLS = [0, 1, 6, 7]  # standardised; the calendar sin/cos columns are already in [-1, 1]


@dataclass
class FeatureStore:
    basket_feats: dict[str, np.ndarray]  # invoice -> (8,) float32, normalised
    quantities: dict[
        str, tuple[np.ndarray, np.ndarray]
    ]  # invoice -> (catalogue idx, quantity), catalogue items only
    customer_index: dict[str, int]  # TRAIN customers -> 1..n (0 = unseen)
    country_index: dict[str, int]  # TRAIN countries -> 1..n (0 = unknown)
    example_arrays: dict[str, dict[str, np.ndarray]]  # split -> customer_idx, country_idx, customer_feats
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_customers(self) -> int:
        return len(self.customer_index)

    @property
    def n_countries(self) -> int:
        return len(self.country_index)


def basket_feature_matrix(baskets: pd.DataFrame) -> tuple[pd.Index, np.ndarray]:
    """Raw (un-normalised) 8 basket features per invoice, from the basket's own data and the PREVIOUS basket.

    0 log1p(total quantity)  1 log1p(days since the customer's previous basket, 0 for the first)
    2,3 sin/cos(month)       4,5 sin/cos(weekday)    6 log1p(distinct products)   7 log1p(max(total value, 0))
    """
    b = baskets.sort_values(["CustomerID", "timestamp", "InvoiceNo"])
    gap = b.groupby("CustomerID")["timestamp"].diff().dt.total_seconds().div(86400.0).fillna(0.0)
    month, weekday = b["timestamp"].dt.month.to_numpy(), b["timestamp"].dt.day_of_week.to_numpy()
    feats = np.column_stack(
        [
            np.log1p(b["total_quantity"].clip(lower=0).to_numpy()),
            np.log1p(gap.to_numpy()),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
            np.log1p(b["n_items"].to_numpy()),
            np.log1p(b["total_value"].clip(lower=0).to_numpy()),
        ]
    )
    return pd.Index(b["InvoiceNo"]), feats


def _standardise(
    raw: np.ndarray, fit_rows: np.ndarray, cols: list[int]
) -> tuple[np.ndarray, dict[str, list[float]]]:
    out = raw.copy()
    mean = raw[fit_rows][:, cols].mean(axis=0)
    std = raw[fit_rows][:, cols].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    out[:, cols] = (raw[:, cols] - mean) / std
    return out, {"mean": mean.tolist(), "std": std.tolist()}


def customer_feature_rows(
    examples: pd.DataFrame, baskets: pd.DataFrame, recency_reference: str
) -> tuple[np.ndarray, list[str]]:
    """Raw 6 customer features per example, from baskets strictly before the target.

    0 log1p(#past baskets)  1 log1p(mean distinct products)  2 log1p(mean quantity)  3 log1p(mean value)
    4 log1p(days from the last past basket to the target time, or 0)  5 log1p(mean days between past baskets)
    Returns the matrix and the country of each example's most recent past basket.
    """
    b = baskets.sort_values(["CustomerID", "timestamp", "InvoiceNo"])
    per_customer = {}
    for cust, g in b.groupby("CustomerID"):
        per_customer[cust] = (
            g["timestamp"].to_numpy(),
            np.concatenate([[0.0], np.cumsum(g["n_items"].to_numpy(dtype=float))]),
            np.concatenate([[0.0], np.cumsum(g["total_quantity"].clip(lower=0).to_numpy(dtype=float))]),
            np.concatenate([[0.0], np.cumsum(g["total_value"].clip(lower=0).to_numpy(dtype=float))]),
            g["Country"].to_numpy(),
        )
    rows = np.zeros((len(examples), 6))
    countries: list[str] = []
    for i, (cust, t) in enumerate(zip(examples["CustomerID"], examples["target_timestamp"], strict=True)):
        stamps, c_items, c_qty, c_val, country = per_customer[cust]
        n = int(np.searchsorted(stamps, np.datetime64(t), side="left"))  # strictly earlier baskets
        assert n >= 1, "every example has at least one past basket"
        days_since_last = (np.datetime64(t) - stamps[n - 1]) / np.timedelta64(1, "D")
        interval = (stamps[n - 1] - stamps[0]) / np.timedelta64(1, "D") / (n - 1) if n > 1 else 0.0
        rows[i] = [
            np.log1p(n),
            np.log1p(c_items[n] / n),
            np.log1p(c_qty[n] / n),
            np.log1p(c_val[n] / n),
            np.log1p(days_since_last) if recency_reference == "target_time" else 0.0,
            np.log1p(interval),
        ]
        countries.append(country[n - 1])
    return rows, countries


def build_feature_store(task: Task, baskets: pd.DataFrame, feat_cfg: dict[str, Any]) -> FeatureStore:
    """Compute all features once. Normalisers use TRAIN baskets and TRAIN examples only."""
    invoices, raw = basket_feature_matrix(baskets)
    split_of = baskets.set_index("InvoiceNo")["split"].reindex(invoices).to_numpy()
    norm, basket_stats = _standardise(raw, split_of == "train", BASKET_LOG_COLS)
    basket_feats = {inv: norm[i].astype(np.float32) for i, inv in enumerate(invoices)}

    quantities = {}
    vocab = {s: i for i, s in enumerate(task.items)}
    for inv, its, qty in zip(baskets["InvoiceNo"], baskets["items"], baskets["quantities"], strict=True):
        pairs = [(vocab[s], q) for s, q in zip(its, qty, strict=True) if s in vocab]
        quantities[inv] = (
            np.array([p[0] for p in pairs], dtype=np.int64),
            np.array([p[1] for p in pairs], dtype=np.float32),
        )

    train_b = baskets[baskets["split"] == "train"]
    customer_index = {c: i + 1 for i, c in enumerate(sorted(train_b["CustomerID"].unique()))}
    country_index = {c: i + 1 for i, c in enumerate(sorted(train_b["Country"].unique()))}

    raw_rows, countries = {}, {}
    for split, ex in task.examples.items():
        raw_rows[split], countries[split] = customer_feature_rows(ex, baskets, feat_cfg["recency_reference"])
    mean = raw_rows["train"].mean(axis=0)
    std = np.where(raw_rows["train"].std(axis=0) < 1e-8, 1.0, raw_rows["train"].std(axis=0))
    arrays = {}
    for split, ex in task.examples.items():
        arrays[split] = {
            "customer_idx": np.array([customer_index.get(c, 0) for c in ex["CustomerID"]], dtype=np.int64),
            "country_idx": np.array([country_index.get(c, 0) for c in countries[split]], dtype=np.int64),
            "customer_feats": ((raw_rows[split] - mean) / std).astype(np.float32),
        }
    stats = {"basket": basket_stats, "customer": {"mean": mean.tolist(), "std": std.tolist()}}
    return FeatureStore(basket_feats, quantities, customer_index, country_index, arrays, stats)
