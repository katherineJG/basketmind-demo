"""Repeat purchases versus new items: where do each model's top-10 hits come from?

A target item is a REPEAT if it is in any of the customer's earlier baskets (the same history the `exclude_history`
policy removes), and NEW otherwise. This is ANALYSIS of finished models from their saved top-10 lists (policy `full`);
nothing here feeds back into model selection. Recall@10 splits exactly into the part earned on repeat targets and the
part earned on new targets, so the two contributions always add up to the headline number.
"""

import numpy as np
import pandas as pd

from basketmind.evaluation.stats import _boot_means, _cluster_sums, cluster_bootstrap, paired_bootstrap


def target_kinds(task, split: str = "test") -> pd.DataFrame:
    """Per example (index = example_id): the repeat and new target items, and the history size."""
    ex = task.examples[split]
    rows = []
    for tgt, hist in zip(ex["target_idx"], ex["history_invoices"], strict=True):
        seen = set(np.concatenate([task.basket_items[i] for i in hist]).tolist()) if len(hist) else set()
        tgt = set(tgt.tolist())
        rows.append({"repeat": tgt & seen, "new": tgt - seen, "history": seen})
    out = pd.DataFrame(rows, index=pd.RangeIndex(len(ex), name="example_id"))
    out["n_repeat"], out["n_new"] = out["repeat"].map(len), out["new"].map(len)
    return out


def repeat_columns(rows: pd.DataFrame, kinds: pd.DataFrame) -> pd.DataFrame:
    """Add the repeat/new breakdown to per-example rows (policy `full` only; `top` is the top-10 list).

    hits_repeat / hits_new  top-10 items that are repeat / new targets
    slots_history           top-10 items that are in the customer's history (repeat candidates)
    contrib_repeat, contrib_new   hits divided by ALL targets, so they add up to recall@10
    cond_repeat, cond_new   share of that kind of target found (NaN when the example has none of that kind)
    """
    rows = rows[rows["policy"] == "full"].copy()
    hits_r, hits_n, slots = [], [], []
    for eid, top in zip(rows["example_id"], rows["top"], strict=True):
        k = kinds.loc[eid]
        top = set(np.asarray(top).tolist())
        hits_r.append(len(top & k["repeat"]))
        hits_n.append(len(top & k["new"]))
        slots.append(len(top & k["history"]))
    rows["hits_repeat"], rows["hits_new"], rows["slots_history"] = hits_r, hits_n, slots
    n_rep, n_new = (kinds[c].reindex(rows["example_id"]).to_numpy() for c in ("n_repeat", "n_new"))
    rows["contrib_repeat"] = rows["hits_repeat"] / (n_rep + n_new)
    rows["contrib_new"] = rows["hits_new"] / (n_rep + n_new)
    rows["cond_repeat"] = np.where(n_rep > 0, rows["hits_repeat"] / np.maximum(n_rep, 1), np.nan)
    rows["cond_new"] = np.where(n_new > 0, rows["hits_new"] / np.maximum(n_new, 1), np.nan)
    check = rows["contrib_repeat"] + rows["contrib_new"] - rows["recall@10"]
    if check.abs().max() > 1e-5:
        raise ValueError(
            f"repeat + new contributions do not add up to recall@10 (max gap {check.abs().max():.2g})"
        )
    return rows


COLUMNS = [
    "hits_repeat",
    "hits_new",
    "slots_history",
    "contrib_repeat",
    "contrib_new",
    "cond_repeat",
    "cond_new",
]


def seed_mean(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, example): the expected value for a model trained with a random seed."""
    return rows.groupby(["model", "example_id"], as_index=False)[COLUMNS].mean()


def interval(avg: pd.DataFrame, model: str, column: str, customers: pd.Series, n_boot: int = 2000) -> dict:
    """Mean of `column` for `model` with a customer-level interval (examples where it is NaN are left out)."""
    d = avg[avg["model"] == model].dropna(subset=[column]).sort_values("example_id")
    return cluster_bootstrap(d[column].to_numpy(), customers.reindex(d["example_id"]).to_numpy(), n_boot)


def slot_precision(avg: pd.DataFrame, model: str, customers: pd.Series, n_boot: int = 2000):
    """Hit rate of the top-10 slots filled by history items, and of those filled by non-history items.

    Returned as ({'mean','lo','hi'} for history slots, same for new slots). A ratio of sums with customers resampled.
    """
    d = avg[avg["model"] == model].sort_values("example_id")
    hist, new = d["slots_history"].to_numpy(), 10 - d["slots_history"].to_numpy()
    cl = customers.reindex(d["example_id"]).to_numpy()
    out = []
    for hits, slots in ((d["hits_repeat"].to_numpy(), hist), (d["hits_new"].to_numpy(), new)):
        num, _ = _cluster_sums(hits, cl)
        den, _ = _cluster_sums(slots, cl)
        boot = _boot_means(num, den, n_boot, 0)
        lo, hi = np.nanquantile(boot, [0.025, 0.975])  # a resample with no such slots at all gives 0/0
        out.append({"mean": float(num.sum() / den.sum()), "lo": float(lo), "hi": float(hi)})
    return tuple(out)


def paired(avg: pd.DataFrame, a: str, b: str, column: str, customers: pd.Series, n_boot: int = 2000) -> dict:
    """`column` of model a minus model b on the same examples, with a customer-level interval."""
    wide = avg.pivot(index="example_id", columns="model", values=column)[[a, b]].dropna()
    return paired_bootstrap(
        wide[a].to_numpy(), wide[b].to_numpy(), customers.reindex(wide.index).to_numpy(), n_boot
    )


NAMES = {
    "item_item_cf": "Item-item CF",
    "gru": "Tuned GRU",
    "abl_graph_customer": "BasketMind (graph + customer)",
    "abl_full": "BasketMind (full hybrid)",
    "matrix_factorisation": "Matrix factorisation",
    "popularity": "Popularity",
    "abl_graph_customer_repeat": "BasketMind + repeat score",
    "personal_frequency": "Personal frequency",
    "personal_frequency_cf": "Personal frequency + CF",
}
PAIRS = [  # (A, B): the reported difference is A minus B
    ("abl_graph_customer", "item_item_cf"),
    ("abl_graph_customer", "gru"),
    ("abl_graph_customer", "personal_frequency"),
    ("abl_graph_customer_repeat", "abl_graph_customer"),
    ("abl_graph_customer_repeat", "personal_frequency"),
    ("abl_graph_customer_repeat", "personal_frequency_cf"),
    ("personal_frequency_cf", "item_item_cf"),
    ("personal_frequency_cf", "personal_frequency"),
]


def _ci(r: dict) -> str:
    return f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def _md(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


def render(avg: pd.DataFrame, kinds: pd.DataFrame, customers: pd.Series, n_boot: int = 2000) -> str:
    """The repeat-purchase report: how much of the test set is repeats, and where each model's hits come from."""
    models = [m for m in NAMES if m in set(avg["model"])]
    n, n_rep, n_new = len(kinds), int(kinds["n_repeat"].sum()), int(kinds["n_new"].sum())
    with_repeat = float((kinds["n_repeat"] > 0).mean())
    recall, cond, comp = [], [], []
    for m in models:
        d = avg[avg["model"] == m]
        total = float((d["contrib_repeat"] + d["contrib_new"]).mean())
        c_rep, c_new = (
            _ci(interval(avg, m, col, customers, n_boot)) for col in ("contrib_repeat", "contrib_new")
        )
        recall.append([NAMES[m], f"{total:.3f}", c_rep, c_new])
        k_rep, k_new = (_ci(interval(avg, m, col, customers, n_boot)) for col in ("cond_repeat", "cond_new"))
        cond.append([NAMES[m], k_rep, k_new])
        hist, new = slot_precision(avg, m, customers, n_boot)
        comp.append([NAMES[m], f"{d['slots_history'].mean() / 10:.0%}", _ci(hist), _ci(new)])
    pairs = []
    for a, b in PAIRS:
        if a in models and b in models:
            for col, label in (
                ("contrib_repeat", "from repeat targets"),
                ("contrib_new", "from new targets"),
            ):
                r = paired(avg, a, b, col, customers, n_boot)
                reading = "gap excludes zero" if r["lo"] > 0 or r["hi"] < 0 else "inconclusive"
                diff = f"{r['mean_diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"
                pairs.append([f"{NAMES[a]} minus {NAMES[b]}", label, diff, reading])
    return f"""# Repeat purchases versus new items

Generated by `scripts/analyse_repeats.py` from the saved per-example test results. Policy `full`, top-10 lists, test split.
A target item is a **repeat** if the customer bought it in any earlier basket, and **new** otherwise.

## How much of the test set is repeats

{n:,} test examples. Of {n_rep + n_new:,} target items (in the catalogue), {n_rep:,} ({n_rep / (n_rep + n_new):.1%}) are repeats and
{n_new:,} are new. {with_repeat:.1%} of examples contain at least one repeat.

## Where Recall@10 comes from

Recall@10 is split exactly into the part earned on repeat targets and the part earned on new targets (each is hits divided by
all targets of that example), so the two right-hand columns add up to the total. Intervals resample customers, 95%.

{_md(["Model", "Recall@10", "from repeat targets", "from new targets"], recall)}

## How many of each kind of target is found

Among examples that have at least one target of that kind: the share of those targets that appear in the top 10.

{_md(["Model", "repeat targets found", "new targets found"], cond)}

## What the top-10 list is made of

Share of the ten slots filled with items from the customer's own history, and how often a slot of each kind is a hit.

{_md(["Model", "slots on history items", "hit rate of history slots", "hit rate of other slots"], comp)}

## Paired differences

Difference in each contribution on the same examples (A minus B). Intervals resample customers. With several comparisons, rely on
the intervals; p-values are not corrected for multiple testing.

{_md(["Comparison", "Part", "Difference [95% CI]", "Reading"], pairs)}
"""
