import json

import numpy as np
import pandas as pd
import pytest
from tests.unit.test_cancellation_labels import REGEX, scenario
from tests.unit.test_rigour import synthetic

from basketmind.inference import analytics


def test_cancellation_payload_is_json_ready_and_consistent_with_the_analysis():
    purchases, cancellations = scenario()
    out = analytics.cancellation_payload(purchases, cancellations, REGEX, {"11111": "WIDGET"})
    json.dumps(out)  # must serialise: this is what the API returns
    assert out["cancellation_lines"] == 3 and out["purchase_lines"] == 3
    assert set(out["horizons"]) == {"30", "90"}  # string keys
    assert sum(m["cancellation_lines"] for m in out["monthly"]) == 3
    assert sum(m["purchase_lines"] for m in out["monthly"]) == 3
    top = out["top_cancelled"][0]
    assert top["product_id"] == "11111" and top["description"] == "WIDGET" and top["cancellation_lines"] == 2
    assert top["share_of_lines"] == pytest.approx(2 / (2 + 2))  # 2 cancellations against 2 purchase lines
    assert all(0 <= m["cancellation_rate"] <= 1 for m in out["monthly"])


def test_personal_frequency_is_named_and_compared_with_the_models_it_beat():
    rows, slices = synthetic()
    pf = rows[rows["model"] == "item_item_cf"].assign(model="personal_frequency")
    for metric in ("recall@10", "ndcg@10", "mrr"):
        pf[metric] = pf[metric] + 0.1  # a known gap over item-item CF
    out = analytics.performance_payload(
        pd.concat([rows, pf], ignore_index=True),
        slices,
        {},
        np.arange(50, 0, -1).astype(float),
        50,
        n_boot=100,
    )
    by_id = {m["id"]: m for m in out["models"]}
    assert by_id["personal_frequency"]["name"] == "Personal frequency"
    assert by_id["personal_frequency"]["group"] == "other" and not by_id["personal_frequency"]["selected"]
    gap = next(
        p
        for p in out["paired"]
        if (p["a"], p["b"], p["metric"]) == ("item_item_cf", "personal_frequency", "recall@10")
        and not p["new_items"]
    )
    assert gap["mean_diff"] == pytest.approx(-0.1, abs=0.01) and gap["hi"] < 0  # A minus B: CF is below it
    assert ("gru", "personal_frequency") in {(p["a"], p["b"]) for p in out["paired"]}
    json.dumps(out)


def test_performance_payload_has_intervals_pairs_slices_and_selection():
    rows, slices = synthetic()
    validation = {"gru": {"recall@10": 0.3, "std": 0.01}}
    out = analytics.performance_payload(
        rows, slices, validation, np.arange(50, 0, -1).astype(float), 50, n_boot=100
    )
    json.dumps(out)
    by_id = {m["id"]: m for m in out["models"]}
    assert set(by_id) == {"item_item_cf", "gru", "popularity"}  # only models present in the data
    cf = by_id["item_item_cf"]["test"]["recall@10"]
    assert cf["lo"] < cf["mean"] < cf["hi"] and by_id["item_item_cf"]["group"] == "other"
    assert by_id["gru"]["group"] == "deep" and by_id["gru"]["validation"] == {"recall@10": 0.3, "std": 0.01}
    assert by_id["gru"]["seed_std_recall10"] > 0 and 0 < by_id["gru"]["coverage"] <= 1
    assert by_id["popularity"]["test_new_items"]["recall@10"]["mean"] < cf["mean"]
    assert not any(
        m["selected"] for m in out["models"]
    )  # the selected variant is absent from this synthetic data
    assert all(
        {"a", "b", "metric", "new_items", "mean_diff", "lo", "hi", "p"} <= set(p) for p in out["paired"]
    )
    assert {s["slice"] for s in out["slices"]["history_bucket"]} == {"1", "2-3", "4-10", "11+"}
    assert {s["slice"] for s in out["slices"]["cold_start"]} == {"True", "False"}
    assert out["meta"]["n_test_examples"] == 400
