import numpy as np
import pandas as pd
import pytest

from basketmind.evaluation.stats import cluster_bootstrap, paired_bootstrap, seed_average


def test_mean_is_the_plain_mean_and_the_interval_contains_it():
    rng = np.random.default_rng(0)
    values = rng.normal(0.3, 1.0, 1000)
    r = cluster_bootstrap(values, np.arange(1000), n_boot=500)
    assert r["mean"] == pytest.approx(values.mean()) and r["lo"] < r["mean"] < r["hi"]
    assert r["n"] == 1000 and r["n_customers"] == 1000
    assert r["hi"] - r["lo"] == pytest.approx(
        2 * 1.96 / np.sqrt(1000), rel=0.25
    )  # close to the textbook interval


def test_correlated_examples_widen_the_interval():
    rng = np.random.default_rng(1)
    customer_effect = np.repeat(rng.normal(0, 1, 100), 10)  # 100 customers x 10 near-identical examples
    values = customer_effect + rng.normal(0, 0.05, 1000)
    clusters = np.repeat(np.arange(100), 10)
    by_customer = cluster_bootstrap(values, clusters, n_boot=500)
    naive = cluster_bootstrap(values, np.arange(1000), n_boot=500)  # treats every example as independent
    assert (by_customer["hi"] - by_customer["lo"]) > 2.5 * (naive["hi"] - naive["lo"])
    assert by_customer["n_customers"] == 100


def test_paired_difference_detects_a_real_gap_and_ignores_none():
    rng = np.random.default_rng(2)
    base = rng.uniform(0, 1, 2000)
    clusters = rng.integers(0, 400, 2000)
    better = paired_bootstrap(base + 0.05, base, clusters, n_boot=500)
    assert better["mean_diff"] == pytest.approx(0.05) and better["lo"] > 0 and better["p"] < 0.01
    same = paired_bootstrap(base, base, clusters, n_boot=500)
    assert same["mean_diff"] == 0.0 and same["lo"] == same["hi"] == 0.0 and same["p"] == 1.0


def test_a_noisy_small_difference_is_not_called_significant():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.5, 0.3, 300), rng.normal(0.5, 0.3, 300)
    r = paired_bootstrap(a, b, np.arange(300), n_boot=1000)
    assert r["lo"] < 0 < r["hi"] and r["p"] > 0.05


def test_swapping_the_models_flips_the_sign_only():
    rng = np.random.default_rng(4)
    a, b, c = rng.normal(0.6, 0.2, 500), rng.normal(0.5, 0.2, 500), rng.integers(0, 100, 500)
    ab, ba = paired_bootstrap(a, b, c, n_boot=500), paired_bootstrap(b, a, c, n_boot=500)
    assert ab["mean_diff"] == pytest.approx(-ba["mean_diff"]) and ab["lo"] == pytest.approx(-ba["hi"])


def test_bootstrap_is_reproducible_for_a_seed():
    v, c = np.random.default_rng(5).normal(size=200), np.arange(200) % 20
    assert cluster_bootstrap(v, c, n_boot=300, seed=9) == cluster_bootstrap(v, c, n_boot=300, seed=9)


def test_seed_average_means_over_seeds_per_example():
    rows = pd.DataFrame(
        {
            "model": ["m"] * 4 + ["det"] * 2,
            "policy": ["full"] * 6,
            "seed": [1, 2, 1, 2, 1, 1],
            "example_id": [0, 0, 1, 1, 0, 1],
            "mrr": [0.2, 0.4, 1.0, 0.0, 0.5, 0.7],
        }
    )
    out = seed_average(rows, ["mrr"]).set_index(["model", "example_id"])["mrr"]
    assert out[("m", 0)] == pytest.approx(0.3) and out[("m", 1)] == pytest.approx(0.5)
    assert out[("det", 0)] == 0.5 and out[("det", 1)] == 0.7
