import numpy as np
import pandas as pd
import pytest

from basketmind.evaluation import rigour
from basketmind.evaluation.slices import HEAD_LABELS


def synthetic(n=400, seed=0):
    """Per-example rows for three models over n examples (2 seeds for 'gru'), with known true gaps."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 0.6, n)
    spec = {"item_item_cf": (base + 0.06, [0]), "gru": (base, [1, 2]), "popularity": (base - 0.15, [0])}
    frames = []
    for model, (mean, seeds) in spec.items():
        for s in seeds:
            for policy in ("full", "exclude_history"):
                v = np.clip(mean + rng.normal(0, 0.02, n), 0, 1)
                f = pd.DataFrame({"model": model, "seed": s, "policy": policy, "example_id": np.arange(n)})
                for m in rigour.METRICS:
                    f[m] = v
                f["top"] = list(rng.integers(0, 50, (n, 10)))
                frames.append(f)
    rows = pd.concat(frames, ignore_index=True)
    slices = pd.DataFrame(
        {
            "customer": rng.integers(0, 80, n),
            "cold_start": rng.random(n) < 0.3,
            "history_bucket": rng.choice(["1", "2-3", "4-10", "11+"], n),
            "country_group": rng.choice(["United Kingdom", "Other"], n),
            "month": rng.choice(["2011-07", "2011-08"], n),
            "target_bucket": rng.choice(HEAD_LABELS, n),
        },
        index=pd.RangeIndex(n, name="example_id"),
    )
    return rows, slices


def test_overall_intervals_contain_the_mean_and_order_the_models():
    rows, slices = synthetic()
    avg = rigour.prepare(rows)
    ov = rigour.overall(avg, slices, "full", "recall@10", n_boot=300)
    assert (ov["lo"] < ov["mean"]).all() and (ov["mean"] < ov["hi"]).all()
    assert ov.loc["item_item_cf", "mean"] > ov.loc["gru", "mean"] > ov.loc["popularity", "mean"]
    assert ov.loc["gru", "n"] == 400


def test_seed_std_is_zero_for_a_single_seed_and_positive_otherwise():
    rows, _ = synthetic()
    sd = rigour.seed_std(rows, "full", "recall@10")
    assert sd["item_item_cf"] == 0.0 and sd["gru"] > 0


def test_paired_table_has_the_expected_sign_and_skips_missing_models():
    rows, slices = synthetic()
    t = rigour.paired(rigour.prepare(rows), slices, "full", "recall@10", n_boot=300)
    assert list(zip(t["a"], t["b"], strict=True)) == [
        ("gru", "item_item_cf")
    ]  # only pairs whose models exist
    r = t.iloc[0]
    assert r["mean_diff"] == pytest.approx(-0.06, abs=0.01) and r["hi"] < 0 and r["p"] < 0.05


def test_slice_table_covers_every_slice_value_and_model():
    rows, slices = synthetic()
    t = rigour.by_slice(rigour.prepare(rows), slices, "history_bucket", "full", "recall@10", n_boot=100)
    assert set(t["slice"]) == {"1", "2-3", "4-10", "11+"} and set(t["model"]) == {
        "item_item_cf",
        "gru",
        "popularity",
    }
    assert t.groupby("slice")["n"].first().sum() == 400  # the slices partition the examples
    t2 = rigour.by_slice(rigour.prepare(rows), slices, "target_bucket", "full", "recall@10", n_boot=100)
    assert list(dict.fromkeys(t2["slice"])) == [
        x for x in HEAD_LABELS if x in set(t2["slice"])
    ]  # fixed order


def test_paired_by_slice_restricts_each_comparison_to_its_slice():
    rows, slices = synthetic()
    t = rigour.paired_by_slice(
        rigour.prepare(rows), slices, "history_bucket", "full", "recall@10", n_boot=100
    )
    pairs = {(a, b) for a, b in zip(t["a"], t["b"], strict=True)}
    assert pairs == {
        ("gru", "item_item_cf")
    }  # the only configured pair whose models exist in the synthetic data
    assert t["n"].sum() == 400 and set(t["slice"]) == {"1", "2-3", "4-10", "11+"}
    assert (t["mean_diff"] < 0).all()  # in every slice the GRU is the weaker model by construction
    cold = rigour.paired_by_slice(rigour.prepare(rows), slices, "cold_start", "full", "recall@10", n_boot=100)
    assert set(cold["slice"]) == {False, True}


def test_new_items_policy_uses_only_its_own_rows():
    rows, slices = synthetic()
    drop = rows["policy"].eq("exclude_history") & rows["example_id"].lt(
        100
    )  # pretend 100 examples were skipped
    ov = rigour.overall(rigour.prepare(rows[~drop]), slices, "exclude_history", "recall@10", n_boot=100)
    assert ov["n"].eq(300).all()


def test_full_report_renders_every_section_without_nan():
    rows, slices = synthetic()
    consistency = {
        "rows": [
            {"model": m, "seed": s, "now": 0.31, "before": 0.30, "diff": 0.01}
            for m, s in (("item_item_cf", 0), ("gru", 1), ("gru", 2), ("popularity", 0))
        ]
    }
    md = rigour.render(rows, slices, consistency, np.arange(50, 0, -1).astype(float), 50, n_boot=100)
    for heading in (
        "Consistency with the reported numbers",
        "95% confidence intervals",
        "New items only",
        "Paired comparisons",
        "by slice",
        "Popularity bias",
    ):
        assert heading in md, heading
    assert "nan" not in md.lower() and "Tuned GRU minus Item-item CF" in md
