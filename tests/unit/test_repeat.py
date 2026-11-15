import numpy as np
import pandas as pd
import pytest

from basketmind.evaluation import repeat


def kinds_frame():
    """Example 0: repeats {1, 2}, new {3}. Example 1: no repeats, new {7, 8}."""
    kinds = pd.DataFrame(
        {
            "repeat": [{1, 2}, set()],
            "new": [{3}, {7, 8}],
            "history": [{1, 2, 5}, {4}],
        },
        index=pd.RangeIndex(2, name="example_id"),
    )
    kinds["n_repeat"], kinds["n_new"] = kinds["repeat"].map(len), kinds["new"].map(len)
    return kinds


def rows_frame(recall):
    top0 = [1, 3, 5, 9, 10, 11, 12, 13, 14, 15]  # hits 1 (repeat) and 3 (new); 5 is a history miss
    top1 = [7, 20, 21, 22, 23, 24, 25, 26, 27, 28]  # hits item 7 (new)
    return pd.DataFrame(
        {
            "example_id": [0, 1],
            "top": [np.array(top0), np.array(top1)],
            "model": "m",
            "seed": 0,
            "policy": "full",
            "recall@10": recall,
        }
    )


def test_hits_split_into_repeat_and_new_and_add_up_to_recall():
    out = repeat.repeat_columns(rows_frame([2 / 3, 1 / 2]), kinds_frame())
    assert out["hits_repeat"].tolist() == [1, 0]
    assert out["hits_new"].tolist() == [1, 1]
    assert out["slots_history"].tolist() == [2, 0]  # items 1 and 5 are in example 0's history
    assert out["contrib_repeat"].tolist() == pytest.approx([1 / 3, 0])
    assert (out["contrib_repeat"] + out["contrib_new"]).tolist() == pytest.approx([2 / 3, 1 / 2])
    assert np.isnan(out["cond_repeat"].iloc[1])  # example 1 has no repeat target
    assert out["cond_new"].tolist() == pytest.approx([1.0, 0.5])


def test_a_recall_that_does_not_match_the_decomposition_is_an_error():
    with pytest.raises(ValueError, match="do not add up"):
        repeat.repeat_columns(rows_frame([0.9, 0.5]), kinds_frame())


def test_only_the_full_policy_is_decomposed():
    rows = rows_frame([2 / 3, 1 / 2])
    rows = pd.concat([rows, rows.assign(policy="exclude_history")], ignore_index=True)
    assert len(repeat.repeat_columns(rows, kinds_frame())) == 2


def test_slot_precision_is_a_ratio_of_sums_inside_its_interval():
    out = repeat.repeat_columns(rows_frame([2 / 3, 1 / 2]), kinds_frame())
    avg = repeat.seed_mean(out)
    hist, new = repeat.slot_precision(avg, "m", pd.Series([10, 11], name="c"), n_boot=200)
    assert hist["mean"] == pytest.approx(1 / 2)  # 1 hit in 2 history slots
    assert new["mean"] == pytest.approx(2 / 18)  # 2 hits in 18 other slots
    for r in (hist, new):
        assert r["lo"] <= r["mean"] <= r["hi"]
