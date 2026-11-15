from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from basketmind.evaluation.slices import HEAD_LABELS, assign_slices, history_bucket, popularity_bias


def fake_task():
    ex = pd.DataFrame(
        {
            "CustomerID": ["a", "b", "c"],
            "history_invoices": [["i1"], ["i2", "i3", "i4"], [f"j{k}" for k in range(12)]],
            "target_timestamp": pd.to_datetime(["2011-07-05", "2011-08-20", "2011-12-01"]),
            "target_idx": [np.array([0, 1]), np.array([5, 6]), np.array([0, 7])],
        }
    )
    pop = np.arange(10, 0, -1).astype(float)  # item 0 is the most popular, item 9 the least
    return SimpleNamespace(examples={"test": ex}, train_customers={"a": 0, "c": 1}, popularity=pop)


def test_history_buckets_cover_every_length():
    assert [history_bucket(n) for n in (1, 2, 3, 4, 10, 11, 50)] == [
        "1",
        "2-3",
        "2-3",
        "4-10",
        "4-10",
        "11+",
        "11+",
    ]


def test_slices_are_assigned_from_the_right_fields(monkeypatch):
    import basketmind.evaluation.slices as sl

    monkeypatch.setattr(sl, "HEAD_N", 2)  # head products: items 0 and 1
    task = fake_task()
    countries = {"i1": "United Kingdom", "i4": "France"}
    countries |= {f"j{k}": "United Kingdom" for k in range(12)}
    s = assign_slices(task, countries | {"i4": "France"}, "test")
    assert s.index.name == "example_id" and len(s) == 3
    assert s["cold_start"].tolist() == [False, True, False]  # "b" has no training basket
    assert s["history_bucket"].tolist() == ["1", "2-3", "11+"]
    assert s["country_group"].tolist() == ["United Kingdom", "Other", "United Kingdom"]  # last history basket
    assert s["month"].tolist() == ["2011-07", "2011-08", "2011-12"]
    assert s["head_share"].tolist() == pytest.approx([1.0, 0.0, 0.5])
    assert s["target_bucket"].tolist() == [HEAD_LABELS[2], HEAD_LABELS[0], HEAD_LABELS[2]]


def test_popularity_bias_extremes():
    pop = np.arange(100, 0, -1).astype(float)  # item 0 most popular
    head_only = popularity_bias(np.tile(np.arange(10), (50, 1)), pop, 100, head_n=10)
    assert head_only["head_share"] == 1.0 and head_only["coverage"] == pytest.approx(0.10)
    assert head_only["mean_percentile"] > 90
    spread = popularity_bias(np.arange(100).reshape(10, 10), pop, 100, head_n=10)
    assert spread["coverage"] == 1.0 and spread["head_share"] == pytest.approx(0.10)
    assert spread["mean_percentile"] == pytest.approx(50.0)
