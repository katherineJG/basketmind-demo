import numpy as np
import pandas as pd
import pytest
import torch

from basketmind.inference import NotFound
from basketmind.utils.device import get_device

DEV = get_device()


@pytest.fixture
def parts(service_parts):
    return service_parts  # (service, training_data, baskets) from tests/conftest.py


def test_service_reproduces_the_training_pipeline_exactly(parts):
    """For every historical example, serving it gives the same logits as the training-time batch."""
    service, data, _ = parts
    checked = 0
    for split, ex in data.task.examples.items():
        for i in range(len(ex)):
            row = ex.iloc[i]
            window = int(data.window[split][i])
            via_service, _ = service.score_customer(
                row["CustomerID"], as_of=row["target_timestamp"], graph=data.snapshot_for(window)
            )
            batch, _ = data.collate(split, np.array([i]))
            with torch.no_grad():
                via_training = service.model(batch.to(DEV), data.snapshot_for(window).to(DEV))[
                    "basket_logits"
                ][0]
            assert torch.allclose(via_service, via_training, atol=1e-5), (split, i)
            checked += 1
    assert checked == sum(len(e) for e in data.task.examples.values()) >= 3


def test_recommendations_are_ranked_by_the_models_own_scores(parts):
    service, _, _ = parts
    recs = service.recommend("1", k=2)
    logits, _ = service.score_customer("1")
    assert [r.rank for r in recs] == [1, 2] and recs[0].score >= recs[1].score
    assert recs[0].score == pytest.approx(float(logits.max()), abs=1e-5)
    assert {r.product_id for r in recs} == {"22000", "85123A"}
    assert all(np.isfinite(r.predicted_quantity) and r.predicted_quantity >= 0 for r in recs)
    assert all(r.description == f"product {r.product_id}" for r in recs)


def test_history_facts_in_the_recommendations_come_from_the_baskets(parts):
    service, _, _ = parts
    by_id = {r.product_id: r for r in service.recommend("1", k=2)}
    assert (
        by_id["22000"].times_bought == 4 and by_id["22000"].last_bought == "2011-08-01"
    )  # baskets 100,101,102,104
    assert by_id["85123A"].times_bought == 2 and by_id["85123A"].in_history


def test_exclude_history_never_returns_owned_items(parts):
    service, _, _ = parts
    only_new = service.recommend("2", k=2, exclude_history=True)  # customer 2 only ever bought 22000
    assert [r.product_id for r in only_new] == ["85123A"] and not only_new[0].in_history
    assert service.recommend("1", k=2, exclude_history=True) == []  # customer 1 owns the whole catalogue


def test_unknown_customers_and_dates_raise_not_found(parts):
    service, _, _ = parts
    with pytest.raises(NotFound):
        service.recommend("nobody")
    with pytest.raises(NotFound, match="no basket before"):
        service.recommend("1", as_of=pd.Timestamp("2009-01-01"))
    with pytest.raises(NotFound):
        service.product("NOPE")
    with pytest.raises(NotFound):
        service.related("NOPE")


def test_customer_views(parts):
    service, _, _ = parts
    listing = service.customers()
    assert [c["customer_id"] for c in listing] == ["1", "2"]  # most baskets first
    assert service.customers(prefix="2")[0]["customer_id"] == "2" and service.n_customers("2") == 1
    profile = service.customer("1")
    assert profile["n_baskets"] == 5 and profile["seen_in_training"] and profile["country"] == "UK"
    hist = service.history("1", limit=2)
    assert [h["invoice"] for h in hist] == ["104", "103"]  # newest first
    assert hist[1]["items"][0]["product_id"] == "85123A" and hist[1]["items"][0]["quantity"] == 4


def test_products_related_and_explanations(parts):
    service, _, _ = parts
    # 22000 is in baskets 100, 101, 102, 104 and 105
    assert service.product("22000")["baskets_containing"] == 5 and service.product("22000")["n_related"] == 1
    rel = service.related("22000")
    assert [r["product_id"] for r in rel] == ["85123A"] and rel[0]["association"] == pytest.approx(0.7)
    ev = service.explain(
        "2", "85123A"
    )  # customer 2 never bought it, but it links to the product they did buy
    assert ev["bought_before"] is False and ev["times_bought"] == 0
    assert [x["product_id"] for x in ev["linked_to_history"]] == ["22000"]
    assert ev["linked_to_history"][0]["association"] == pytest.approx(0.7)
    assert ev["model_rank"] in (1, 2) and 1 <= ev["popularity_rank"] <= 2


def test_overview_counts_match_the_data(parts):
    service, _, baskets = parts
    ov = service.overview()
    assert ov["baskets"] == len(baskets) and ov["customers"] == 2
    assert sum(m["baskets"] for m in ov["monthly"]) == len(baskets)
    assert ov["first_basket"] == str(baskets["timestamp"].min().date())
    assert service.metadata == {"variant": "test"}
