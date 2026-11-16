import numpy as np
import pandas as pd
import pytest

from basketmind.evaluation import quantity as q

CUSTOMERS = np.array(["a", "b"])


def test_metrics_match_hand_computed_values():
    out = q.metrics(np.array([2.0, 4.0]), np.array([3.0, 2.0]), CUSTOMERS, n_boot=50)
    assert out["mae"]["mean"] == pytest.approx(1.5)
    assert out["rmse"]["mean"] == pytest.approx(np.sqrt(2.5))
    assert out["wape"]["mean"] == pytest.approx(3 / 6)  # total absolute error over total real quantity
    assert out["mae_log1p"] == pytest.approx(
        np.mean([abs(np.log1p(3) - np.log1p(2)), abs(np.log1p(2) - np.log1p(4))])
    )


def test_several_seed_columns_are_averaged_error_by_error():
    y = np.array([2.0, 4.0])
    two_seeds = np.array([[3.0, 1.0], [2.0, 4.0]])  # pair 1 is off by 1 and 1; pair 2 by 2 and 0
    assert q.metrics(y, two_seeds, CUSTOMERS, n_boot=50)["mae"]["mean"] == pytest.approx(1.0)


def test_paired_difference_is_negative_when_the_first_model_is_closer():
    y = np.array([2.0, 4.0])
    r = q.paired_abs_error(y, y, np.array([3.0, 2.0]), CUSTOMERS, n_boot=50)
    assert r["mean_diff"] == pytest.approx(-1.5)


def synthetic_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    actual = rng.integers(1, 20, n).astype(float)
    frame = pd.DataFrame(
        {
            "example_id": np.arange(n) // 3,
            "customer": rng.choice(list("abcde"), n),
            "item": rng.integers(0, 5, n),
            "actual": actual,
            "last_qty": np.where(rng.random(n) < 0.5, actual + 1, np.nan),
            "global_median": 6.0,
            "item_median": actual + rng.normal(0, 2, n),
            "last_or_item_median": actual + rng.normal(0, 2, n),
            "head_seed1": actual + rng.normal(0, 3, n),
            "head_seed2": actual + rng.normal(0, 3, n),
        }
    )
    return frame


def test_the_report_has_every_model_and_both_kinds_of_pair():
    md = q.render({"validation": synthetic_frame()}, n_boot=50)
    assert "## Validation" in md and "BasketMind quantity head" in md and "Global median quantity" in md
    assert "products bought before" in md and "Paired differences" in md
    assert q.predictions(synthetic_frame())["head"].shape == (60, 2)
