import numpy as np
import pytest

from basketmind.evaluation import quantity as q
from basketmind.models import BasketMindModel
from basketmind.utils.config import load_config
from basketmind.utils.device import get_device
from basketmind.utils.seed import seed_everything

DEV = get_device()


def test_pairs_are_the_target_products_with_the_last_quantity_from_history(training_data):
    store, ex = training_data.store, training_data.task.examples["test"]
    pairs = q.target_pairs(training_data, "test")
    # an independent, brute-force version of the same lookup
    expected = []
    for eid, row in enumerate(ex.itertuples()):
        items, qty = store.quantities[row.target_invoice]
        for item, actual in zip(items.tolist(), qty.tolist(), strict=True):
            last = np.nan
            for inv in row.history_invoices:
                h_items, h_qty = store.quantities[inv]
                if item in h_items:
                    last = float(h_qty[list(h_items).index(item)])
            expected.append((eid, item, actual, last))
    assert len(pairs) == len(expected) > 0
    assert pairs["example_id"].tolist() == [e[0] for e in expected]
    assert pairs["item"].tolist() == [e[1] for e in expected]
    assert pairs["actual"].tolist() == pytest.approx([e[2] for e in expected])
    assert np.allclose(pairs["last_qty"].to_numpy(), [e[3] for e in expected], equal_nan=True)


def test_baselines_use_training_baskets_only(training_data):
    store = training_data.store
    train = {
        inv for inv in store.quantities if inv in training_data.task.basket_items
    }  # every invoice, as a superset
    first = sorted(train)[:2]
    global_median, item_median = q.fit_baselines(training_data, set(first))
    pooled = np.concatenate([store.quantities[i][1] for i in first])
    assert global_median == pytest.approx(float(np.median(pooled)))
    for invoice in first:
        for item in store.quantities[invoice][0]:
            assert item_median[int(item)] == pytest.approx(
                float(
                    np.median(
                        [
                            store.quantities[i][1][list(store.quantities[i][0]).index(item)]
                            for i in first
                            if item in store.quantities[i][0]
                        ]
                    )
                )
            )
    pairs = q.target_pairs(training_data, "test")
    preds = q.baseline_predictions(pairs, global_median, item_median)
    assert len(preds) == len(pairs) and (preds["item_median"] > 0).all()
    seen = pairs["last_qty"].notna()
    assert (preds.loc[seen, "last_or_item_median"] == pairs.loc[seen, "last_qty"]).all()
    assert (preds.loc[~seen, "last_or_item_median"] == preds.loc[~seen, "item_median"]).all()


def test_head_predictions_cover_every_pair_and_do_not_depend_on_the_batch_size(training_data):
    task, store = training_data.task, training_data.store
    seed_everything(0)
    net = BasketMindModel(load_config("model"), task.n_items, store.n_customers, store.n_countries).to(DEV)
    pairs = q.target_pairs(training_data, "test")
    one = q.model_predictions(net, training_data, "test", pairs, DEV, batch_size=1)
    many = q.model_predictions(net, training_data, "test", pairs, DEV, batch_size=8)
    assert one.shape == (len(pairs),) and np.isfinite(one).all() and (one >= 0).all()
    assert one == pytest.approx(many, abs=1e-4)
