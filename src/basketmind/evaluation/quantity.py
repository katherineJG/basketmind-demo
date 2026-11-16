"""Quantity head: how many units of a product does a customer buy, given that they buy it?

The head scores (customer, product) pairs, so it is evaluated on the products that really are in the target basket
(catalogue products only, like every other metric here). Errors are measured in units, not in the log space the head is
trained in. Baselines are fitted on training baskets only. Nothing here changes a model or feeds back into selection.
"""

import numpy as np
import pandas as pd
import torch

from basketmind.evaluation.stats import _boot_means, _cluster_sums, cluster_bootstrap, paired_bootstrap


def target_pairs(data, split: str) -> pd.DataFrame:
    """One row per (example, target product): the real quantity and the customer's last quantity of it, if any.

    `last_qty` is the quantity in the customer's most recent earlier basket (inside the example's history) that
    contains the product, NaN if they never bought it there.
    """
    store, ex = data.store, data.task.examples[split]
    rows = []
    for eid, (customer, target, history) in enumerate(
        zip(ex["CustomerID"], ex["target_invoice"], ex["history_invoices"], strict=True)
    ):
        seen: dict[int, float] = {}
        for invoice in history:  # oldest first, so later baskets overwrite earlier ones
            items, qty = store.quantities[invoice]
            seen.update(zip(items.tolist(), qty.tolist(), strict=True))
        items, qty = store.quantities[target]
        rows += [
            (eid, customer, i, q, seen.get(i, np.nan))
            for i, q in zip(items.tolist(), qty.tolist(), strict=True)
        ]
    return pd.DataFrame(rows, columns=["example_id", "customer", "item", "actual", "last_qty"])


def fit_baselines(data, train_invoices: set[str]) -> tuple[float, dict[int, float]]:
    """Global median quantity and per-product median quantity, from TRAIN baskets only."""
    per_item: dict[int, list[float]] = {}
    for invoice in train_invoices:
        items, qty = data.store.quantities[invoice]
        for i, q in zip(items.tolist(), qty.tolist(), strict=True):
            per_item.setdefault(i, []).append(q)
    all_q = np.concatenate([q for q in (np.asarray(v) for v in per_item.values())])
    return float(np.median(all_q)), {i: float(np.median(v)) for i, v in per_item.items()}


def baseline_predictions(
    pairs: pd.DataFrame, global_median: float, item_median: dict[int, float]
) -> pd.DataFrame:
    """Predicted quantity per pair for each baseline (same row order as `pairs`)."""
    item = pairs["item"].map(item_median).fillna(global_median)
    return pd.DataFrame(
        {
            "global_median": np.full(len(pairs), global_median),
            "item_median": item.to_numpy(),
            "last_or_item_median": pairs["last_qty"].fillna(item).to_numpy(),
        }
    )


@torch.no_grad()
def model_predictions(
    net: torch.nn.Module, data, split: str, pairs: pd.DataFrame, device, batch_size: int = 64
):
    """The quantity head's prediction in units (expm1 of its log1p output, floored at 0) for every pair."""
    net.eval()
    out = np.zeros(len(pairs))
    groups = pairs.groupby("example_id").indices  # example id -> row positions in `pairs`
    items = pairs["item"].to_numpy()
    for window, idx in data.iter_batches(split, batch_size):
        batch, _ = data.collate(split, idx)
        k = max(len(groups[int(e)]) for e in idx)
        q_items = torch.zeros(len(idx), k, dtype=torch.long)
        for row, e in enumerate(idx):
            pos = groups[int(e)]
            q_items[row, : len(pos)] = torch.as_tensor(items[pos])
        pred = net(batch.to(device), data.snapshot_for(window).to(device), q_items.to(device))["quantity"]
        pred = np.maximum(np.expm1(pred.cpu().numpy()), 0.0)
        for row, e in enumerate(idx):
            pos = groups[int(e)]
            out[pos] = pred[row, : len(pos)]
    return out


def _ratio(num: np.ndarray, den: np.ndarray, clusters: np.ndarray, n_boot: int, root: bool = False) -> dict:
    """num.sum() / den.sum() (optionally square-rooted) with a customer-level interval."""
    n, _ = _cluster_sums(num, clusters)
    d, _ = _cluster_sums(den, clusters)
    boot = _boot_means(n, d, n_boot, 0)
    est, lo, hi = n.sum() / d.sum(), *np.quantile(boot, [0.025, 0.975])
    f = np.sqrt if root else float
    return {"mean": float(f(est)), "lo": float(f(lo)), "hi": float(f(hi))}


def _cols(pred) -> np.ndarray:
    """(pairs, models-or-seeds) predictions; a single column may be passed as a 1-D array."""
    p = np.asarray(pred, dtype=float)
    return p[:, None] if p.ndim == 1 else p


def metrics(actual: np.ndarray, pred, customers: np.ndarray, n_boot: int = 2000) -> dict:
    """MAE, RMSE and WAPE in units (customer-level intervals) and MAE in the log1p space the head is trained in.

    `pred` may have one column per training seed: errors are averaged over seeds for each pair first, as in the
    other per-example analyses, so the result is the expected score of a model trained with a random seed.
    """
    y, p = np.asarray(actual, dtype=float), _cols(pred)
    err = p - y[:, None]
    return {
        "mae": cluster_bootstrap(np.abs(err).mean(1), customers, n_boot),
        "rmse": _ratio((err**2).mean(1), np.ones(len(y)), customers, n_boot, root=True),
        "wape": _ratio(np.abs(err).mean(1), y, customers, n_boot),
        "mae_log1p": float(np.abs(np.log1p(np.maximum(p, 0)) - np.log1p(y)[:, None]).mean()),
    }


def paired_abs_error(actual, pred_a, pred_b, customers, n_boot: int = 2000) -> dict:
    """Absolute error of A minus that of B on the same pairs (negative: A is closer), customer-level interval."""
    y = np.asarray(actual, dtype=float)[:, None]
    return paired_bootstrap(
        np.abs(_cols(pred_a) - y).mean(1), np.abs(_cols(pred_b) - y).mean(1), customers, n_boot
    )


NAMES = {
    "global_median": "Global median quantity",
    "item_median": "Product's median quantity",
    "last_or_item_median": "Customer's last quantity, else product median",
    "head": "BasketMind quantity head",
}


def _ci(r: dict, digits: int = 2) -> str:
    return f"{r['mean']:.{digits}f} [{r['lo']:.{digits}f}, {r['hi']:.{digits}f}]"


def _md(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


def predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each model's predictions; the head has one column per seed."""
    seeds = sorted(c for c in frame.columns if c.startswith("head_seed"))
    out = {m: frame[m].to_numpy() for m in ("global_median", "item_median", "last_or_item_median")}
    out["head"] = frame[seeds].to_numpy()
    return out


def render_split(frame: pd.DataFrame, split: str, n_boot: int = 2000) -> str:
    y, customers = frame["actual"].to_numpy(), frame["customer"].to_numpy()
    preds = predictions(frame)
    repeat = frame["last_qty"].notna().to_numpy()
    q = np.quantile(y, [0.5, 0.9, 0.99])
    overall, kinds = [], []
    for m, name in NAMES.items():
        r = metrics(y, preds[m], customers, n_boot)
        overall.append([name, _ci(r["mae"]), _ci(r["rmse"]), _ci(r["wape"], 3), f"{r['mae_log1p']:.3f}"])
        by = {
            k: metrics(y[mask], preds[m][mask], customers[mask], n_boot)["mae"]
            for k, mask in (("repeat", repeat), ("new", ~repeat))
        }
        kinds.append([name, _ci(by["repeat"]), _ci(by["new"])])
    seed_mae = [float(np.abs(preds["head"][:, s] - y).mean()) for s in range(preds["head"].shape[1])]
    pairs = []
    for base in ("item_median", "last_or_item_median"):
        r = paired_abs_error(y, preds["head"], preds[base], customers, n_boot)
        reading = "gap excludes zero" if r["lo"] > 0 or r["hi"] < 0 else "inconclusive"
        pairs.append(
            [
                f"Head minus {NAMES[base].lower()}",
                f"{r['mean_diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]",
                reading,
            ]
        )
    return f"""## {split.capitalize()}

{len(y):,} (customer, product) pairs from {frame["example_id"].nunique():,} examples and {frame["customer"].nunique():,} customers.
{repeat.mean():.0%} of pairs are products the customer bought before. Real quantities: median {q[0]:.0f}, mean {y.mean():.1f},
90th percentile {q[1]:.0f}, 99th percentile {q[2]:.0f}, maximum {y.max():.0f}. Intervals resample customers (95%).

{_md(["Model", "MAE (units)", "RMSE (units)", "WAPE", "MAE in log1p"], overall)}

The head's numbers average the errors of its {preds["head"].shape[1]} seeds for each pair. Its MAE per seed ranges from
{min(seed_mae):.3f} to {max(seed_mae):.3f}.

### By kind of pair (MAE in units)

{_md(["Model", "products bought before", "new to the customer"], kinds)}

### Paired differences in absolute error (units; negative: the head is closer)

{_md(["Comparison", "Difference [95% CI]", "Reading"], pairs)}
"""


def render(frames: dict[str, pd.DataFrame], n_boot: int = 2000) -> str:
    sections = "\n".join(render_split(f, s, n_boot) for s, f in frames.items())
    return f"""# Quantity head

Generated by `scripts/analyse_quantity.py` from `scripts/evaluate_quantity.py`; do not edit by hand. The quantity head
predicts `log1p(quantity)` for a (customer, product) pair and is trained jointly with the basket head on 16 sampled target
products per example. Here it is scored on every catalogue product in the real target basket, so the question it answers is
*given that the customer buys this product, how many units?* It says nothing about whether they buy it. Errors are in units.

Baselines are fitted on training baskets only: the global median quantity, each product's median quantity, and the
customer's own most recent quantity of the product (inside the example's history) with the product median when they have
never bought it. The model is the selected BasketMind variant (graph + customer), three seeds, evaluated as saved.
Test is evaluated once.

{sections}"""
