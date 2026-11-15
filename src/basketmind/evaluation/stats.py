"""Confidence intervals and paired comparisons that resample whole customers.

Examples from the same customer are not independent (the same person, the same tastes), so a plain bootstrap
over examples would give intervals that are too narrow. Here the unit that is resampled is the customer, with
all of their examples. The intervals describe variation due to WHICH customers are in the test set; variation
from training seeds is reported separately as the standard deviation over seeds.
"""

import numpy as np
import pandas as pd


def _cluster_sums(values: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes, _ = pd.factorize(np.asarray(clusters))
    return np.bincount(codes, weights=np.asarray(values, dtype=float)), np.bincount(codes).astype(float)


def _boot_means(sums: np.ndarray, counts: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def cluster_bootstrap(
    values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> dict[str, float]:
    """Mean of `values` with a percentile confidence interval from resampling customers."""
    sums, counts = _cluster_sums(values, clusters)
    boot = _boot_means(sums, counts, n_boot, seed)
    a = (1 - level) / 2
    return {
        "mean": float(sums.sum() / counts.sum()),
        "lo": float(np.quantile(boot, a)),
        "hi": float(np.quantile(boot, 1 - a)),
        "n": int(counts.sum()),
        "n_customers": len(sums),
    }


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, clusters: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> dict[str, float]:
    """Mean of (a - b) over the same examples, its interval, and a two-sided bootstrap p-value (null: no gap).

    The p-value is the share of resampled mean differences on the other side of zero (doubled, +1 smoothed),
    so it cannot be exactly 0. With several comparisons, read the intervals rather than ranking p-values.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sums, counts = _cluster_sums(d, clusters)
    boot = _boot_means(sums, counts, n_boot, seed)
    mean = float(sums.sum() / counts.sum())
    tail = (np.sum(boot <= 0) if mean > 0 else np.sum(boot >= 0)) + 1
    al = (1 - level) / 2
    return {
        "mean_diff": mean,
        "lo": float(np.quantile(boot, al)),
        "hi": float(np.quantile(boot, 1 - al)),
        "p": float(min(1.0, 2 * tail / (n_boot + 1))),
        "n": int(counts.sum()),
    }


def seed_average(rows: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per-example mean over seeds: one row per (model, policy, example_id).

    The result is the expected per-example score of a model trained with a random seed; deterministic models
    (one seed) pass through unchanged.
    """
    return rows.groupby(["model", "policy", "example_id"], as_index=False)[metrics].mean()
