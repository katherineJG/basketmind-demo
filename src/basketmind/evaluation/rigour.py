"""Tables for the evaluation-rigour report: confidence intervals, paired comparisons, slices, popularity bias."""

import numpy as np
import pandas as pd

from basketmind.evaluation.slices import HEAD_LABELS, popularity_bias
from basketmind.evaluation.stats import cluster_bootstrap, paired_bootstrap, seed_average

NAMES = {
    "item_item_cf": "Item-item CF",
    "gru": "Tuned GRU",
    "abl_graph_customer": "BasketMind (graph + customer)",
    "abl_full": "BasketMind (full hybrid)",
    "matrix_factorisation": "Matrix factorisation",
    "popularity": "Popularity",
}
METRICS = ["recall@5", "recall@10", "recall@20", "precision@10", "ndcg@10", "mrr"]
PAIRS = [  # (A, B): the reported difference is A minus B
    ("abl_graph_customer", "gru"),
    ("abl_graph_customer", "item_item_cf"),
    ("abl_graph_customer", "abl_full"),
    ("abl_graph_customer", "matrix_factorisation"),
    ("gru", "item_item_cf"),
]
SLICES = [
    (
        "cold_start",
        "Customer seen in training?",
        {False: "seen in training", True: "cold start (no training basket)"},
    ),
    ("history_bucket", "Earlier baskets in the history", None),
    ("country_group", "Country of the customer", None),
    ("month", "Month of the target basket", None),
    ("target_bucket", "Share of the target made of the 100 most popular products", None),
]


def prepare(rows: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged per-example metrics: one row per (model, policy, example_id)."""
    return seed_average(rows.drop(columns=["top"]), METRICS)


def wide(avg: pd.DataFrame, policy: str, metric: str) -> pd.DataFrame:
    sub = avg[avg["policy"] == policy]
    return sub.pivot(index="example_id", columns="model", values=metric)


def overall(avg: pd.DataFrame, slices: pd.DataFrame, policy: str, metric: str, n_boot: int) -> pd.DataFrame:
    w = wide(avg, policy, metric)
    clusters = slices.loc[w.index, "customer"].to_numpy()
    rows = [{"model": m, **cluster_bootstrap(w[m].to_numpy(), clusters, n_boot)} for m in w.columns]
    return pd.DataFrame(rows).set_index("model")


def seed_std(rows: pd.DataFrame, policy: str, metric: str) -> pd.Series:
    """Standard deviation over training seeds of each model's mean metric (0 for deterministic models)."""
    per_seed = rows[rows["policy"] == policy].groupby(["model", "seed"])[metric].mean()
    return per_seed.groupby("model").std(ddof=0)


def paired(
    avg: pd.DataFrame, slices: pd.DataFrame, policy: str, metric: str, n_boot: int, pairs=PAIRS
) -> pd.DataFrame:
    w = wide(avg, policy, metric)
    clusters = slices.loc[w.index, "customer"].to_numpy()
    rows = [
        {"a": a, "b": b, **paired_bootstrap(w[a].to_numpy(), w[b].to_numpy(), clusters, n_boot)}
        for a, b in pairs
        if a in w.columns and b in w.columns
    ]
    return pd.DataFrame(rows)


def by_slice(avg: pd.DataFrame, slices: pd.DataFrame, column: str, policy: str, metric: str, n_boot: int):
    """Mean metric and a confidence half-width for every (slice value, model)."""
    w = wide(avg, policy, metric)
    sl = slices.loc[w.index]
    out = []
    for value in sorted(
        sl[column].unique(), key=lambda v: str(v) if column != "target_bucket" else HEAD_LABELS.index(v)
    ):
        mask = (sl[column] == value).to_numpy()
        for m in w.columns:
            r = cluster_bootstrap(w[m].to_numpy()[mask], sl["customer"].to_numpy()[mask], n_boot)
            out.append(
                {"slice": value, "model": m, "mean": r["mean"], "half": (r["hi"] - r["lo"]) / 2, "n": r["n"]}
            )
    return pd.DataFrame(out)


SLICE_PAIRS = [("abl_graph_customer", "item_item_cf"), ("gru", "item_item_cf"), ("abl_graph_customer", "gru")]


def paired_by_slice(
    avg: pd.DataFrame, slices: pd.DataFrame, column: str, policy: str, metric: str, n_boot: int
) -> pd.DataFrame:
    """Paired difference [CI] for the main model pairs inside each value of a slice column."""
    w = wide(avg, policy, metric)
    sl = slices.loc[w.index]
    out = []
    for value in sorted(sl[column].unique(), key=str):
        mask = (sl[column] == value).to_numpy()
        for a, b in SLICE_PAIRS:
            if a in w.columns and b in w.columns:
                r = paired_bootstrap(
                    w[a].to_numpy()[mask], w[b].to_numpy()[mask], sl["customer"].to_numpy()[mask], n_boot
                )
                out.append({"slice": value, "a": a, "b": b, **r})
    return pd.DataFrame(out)


def bias_table(rows: pd.DataFrame, popularity: np.ndarray, n_items: int) -> pd.DataFrame:
    """Popularity-bias measures from the top-10 lists (policy `full`), averaged over seeds."""
    full = rows[rows["policy"] == "full"]
    out = {}
    for (model, _seed), g in full.groupby(["model", "seed"]):
        out.setdefault(model, []).append(popularity_bias(np.stack(g["top"].to_numpy()), popularity, n_items))
    return pd.DataFrame({m: pd.DataFrame(v).mean() for m, v in out.items()}).T


def _ci(r) -> str:
    return f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"


def _md(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


def render(
    rows: pd.DataFrame,
    slices: pd.DataFrame,
    consistency: dict,
    popularity: np.ndarray,
    n_items: int,
    n_boot: int = 2000,
) -> str:
    avg = prepare(rows)
    order = list(NAMES)
    sections = []

    cons = pd.DataFrame(consistency["rows"])
    cons_rows = [
        [NAMES[m], f"{g['before'].mean():.4f}", f"{g['now'].mean():.4f}", f"{g['diff'].abs().max():.4f}"]
        for m, g in cons.groupby("model", sort=False)
    ]
    sections.append(
        "## Consistency with the reported numbers\n\n"
        "Mean test Recall@10 (repeats allowed) in the runs reported earlier and in the runs used here. BasketMind models are "
        "the same checkpoints, so they must match. Baselines were refitted (they are not saved), so small differences are expected.\n\n"
        + _md(
            ["Model", "Reported", "This analysis", "Largest per-seed change"],
            sorted(cons_rows, key=lambda r: order.index(next(k for k, v in NAMES.items() if v == r[0]))),
        )
    )

    ov = {m: overall(avg, slices, "full", m, n_boot) for m in ("recall@10", "ndcg@10", "mrr")}
    sd = seed_std(rows, "full", "recall@10")
    ci_rows = [
        [NAMES[m]] + [_ci(ov[k].loc[m]) for k in ("recall@10", "ndcg@10", "mrr")] + [f"{sd[m]:.4f}"]
        for m in order
        if m in ov["mrr"].index
    ]
    n_ex, n_cust = int(ov["mrr"]["n"].iloc[0]), int(ov["mrr"]["n_customers"].iloc[0])
    sections.append(
        f"## Test results with 95% confidence intervals (repeat purchases allowed)\n\n"
        f"{n_ex:,} test examples from {n_cust:,} customers. Intervals come from a bootstrap that resamples customers with all "
        f"their examples ({n_boot:,} resamples); they reflect which customers happen to be in the test set. The last column is the "
        f"standard deviation of Recall@10 across training seeds (a different source of variation). Per-example scores are averaged "
        f"over the three seeds first.\n\n"
        + _md(["Model", "Recall@10", "NDCG@10", "MRR", "Seed std (Recall@10)"], ci_rows)
    )

    ov_new = overall(avg, slices, "exclude_history", "recall@10", n_boot)
    sections.append(
        "## New items only (the customer's own history removed)\n\n"
        + _md(["Model", "Recall@10"], [[NAMES[m], _ci(ov_new.loc[m])] for m in order if m in ov_new.index])
    )

    pair_rows = []
    for policy, metrics in (("full", ["recall@10", "ndcg@10", "mrr"]), ("exclude_history", ["recall@10"])):
        for metric in metrics:
            for _, r in paired(avg, slices, policy, metric, n_boot).iterrows():
                verdict = "gap excludes zero" if r["lo"] > 0 or r["hi"] < 0 else "inconclusive"
                pair_rows.append(
                    [
                        f"{NAMES[r['a']]} minus {NAMES[r['b']]}",
                        metric + (" (new items)" if policy != "full" else ""),
                        f"{r['mean_diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]",
                        f"{r['p']:.3f}",
                        verdict,
                    ]
                )
    sections.append(
        "## Paired comparisons\n\n"
        "Difference in the per-example score between two models on the same examples, with a customer-level bootstrap interval. "
        "'Gap excludes zero' means the 95% interval does not contain zero. With this many comparisons, rely on the intervals; the "
        "p-values are not corrected for multiple testing.\n\n"
        + _md(["Comparison", "Metric", "Difference [95% CI]", "p", "Reading"], pair_rows)
    )

    slice_md = []
    for column, title, labels in SLICES:
        t = by_slice(avg, slices, column, "full", "recall@10", min(n_boot, 500))
        pivot_rows = []
        for value in t["slice"].unique():
            g = t[t["slice"] == value].set_index("model")
            label = (labels or {}).get(value, str(value))
            pivot_rows.append(
                [f"{label} (n={int(g['n'].iloc[0]):,})"]
                + [f"{g.loc[m, 'mean']:.3f} ±{g.loc[m, 'half']:.3f}" if m in g.index else "-" for m in order]
            )
        slice_md.append(f"### {title}\n\n" + _md(["Slice"] + [NAMES[m] for m in order], pivot_rows))
    sections.append(
        "## Test Recall@10 by slice (repeat purchases allowed)\n\n"
        "Mean Recall@10 with a 95% half-width (customer bootstrap, 500 resamples) per slice. Small slices have wide intervals.\n\n"
        + "\n\n".join(slice_md)
    )

    sp_rows = []
    for column, label_of in (
        ("history_bucket", lambda v: f"{v} earlier baskets"),
        ("cold_start", lambda v: "cold start" if v else "seen in training"),
    ):
        for _, r in paired_by_slice(avg, slices, column, "full", "recall@10", min(n_boot, 1000)).iterrows():
            reading = "gap excludes zero" if r["lo"] > 0 or r["hi"] < 0 else "inconclusive"
            sp_rows.append(
                [
                    f"{label_of(r['slice'])} (n={int(r['n']):,})",
                    f"{NAMES[r['a']]} minus {NAMES[r['b']]}",
                    f"{r['mean_diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]",
                    reading,
                ]
            )
    sections.append(
        "## Paired differences inside slices (Recall@10, repeat purchases allowed)\n\n"
        "The same paired customer-level bootstrap, restricted to one slice at a time (1,000 resamples). Positive means the first "
        "model is better. Several slices and pairs are shown, so treat single 'excludes zero' results with caution.\n\n"
        + _md(["Slice", "Comparison", "Difference [95% CI]", "Reading"], sp_rows)
    )

    bias = bias_table(rows, popularity, n_items)
    sections.append(
        "## Popularity bias and catalogue coverage (top-10 lists, repeat purchases allowed)\n\n"
        "Coverage is the share of the catalogue that is ever recommended; head share is the share of recommendations that are the "
        "100 most popular training products; mean percentile is the average popularity rank of recommended products (100 = most popular).\n\n"
        + _md(
            ["Model", "Coverage", "Head share", "Mean percentile"],
            [
                [
                    NAMES[m],
                    f"{bias.loc[m, 'coverage']:.1%}",
                    f"{bias.loc[m, 'head_share']:.1%}",
                    f"{bias.loc[m, 'mean_percentile']:.0f}",
                ]
                for m in order
                if m in bias.index
            ],
        )
    )
    return (
        "# Evaluation rigour\n\nGenerated by `scripts/analyse_results.py` from per-example test results; do not edit by hand. "
        "This is analysis of finished models: nothing here was used to choose or tune a model.\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
