import numpy as np
import pandas as pd
import pytest

from basketmind.baselines.task import load_task
from basketmind.data.pipeline import run_m1
from basketmind.features.build import basket_feature_matrix, build_feature_store, customer_feature_rows
from basketmind.training.data import build_training_data
from basketmind.utils.config import load_config

FEAT = load_config("features")
GRAPH = load_config("graph")
STAMP = pd.Timestamp


@pytest.fixture
def processed(synthetic_root):
    run_m1(root=synthetic_root)
    return synthetic_root / "data/processed"


@pytest.fixture
def baskets(processed):
    return pd.read_parquet(processed / "baskets.parquet")


def test_basket_features_hand_computed(baskets):
    inv, raw = basket_feature_matrix(baskets)
    row = raw[list(inv).index("102")]  # customer 1, 2010-04-01 09:00, previous basket 101 at 2010-03-01 10:00
    gap = (STAMP("2010-04-01 09:00") - STAMP("2010-03-01 10:00")).total_seconds() / 86400
    expected = [np.log1p(1), np.log1p(gap), np.sin(2 * np.pi * 4 / 12), np.cos(2 * np.pi * 4 / 12)]
    assert np.allclose(row[:4], expected)
    assert np.allclose(row[4:6], [np.sin(2 * np.pi * 3 / 7), np.cos(2 * np.pi * 3 / 7)])  # Thursday = 3
    assert np.allclose(row[6:], [np.log1p(1), np.log1p(5.0)])  # 1 product, value 1 x 5.0
    first = raw[list(inv).index("105")]  # the only basket of customer 2
    assert first[1] == 0.0  # no previous basket: gap 0


def test_customer_features_hand_computed_and_recency_switch(processed, baskets):
    ex = load_task(processed).examples["validation"]  # basket 103, history 100/101/102
    raw, countries = customer_feature_rows(ex, baskets, "target_time")
    r = raw[0]
    interval = (STAMP("2010-04-01 09:00") - STAMP("2010-03-01 10:00")).total_seconds() / 86400 / 2
    recency = (STAMP("2011-02-01 09:00") - STAMP("2010-04-01 09:00")).days
    assert np.allclose(r[0], np.log1p(3))  # three past baskets
    assert np.allclose(r[1], np.log1p((2 + 1 + 1) / 3))  # mean distinct products
    assert np.allclose(r[2], np.log1p((6 + 1 + 1) / 3))  # mean quantity (5+1, 1, 1)
    assert np.allclose(r[3], np.log1p((15 + 5 + 5) / 3))  # mean value
    assert np.allclose(r[4], np.log1p(recency)) and np.allclose(r[5], np.log1p(interval))
    assert countries == ["UK"]
    assert customer_feature_rows(ex, baskets, "none")[0][0, 4] == 0.0  # time-blind variant


def test_customer_features_ignore_baskets_after_the_target(processed, baskets):
    ex = load_task(processed).examples["validation"]
    base, _ = customer_feature_rows(ex, baskets, "target_time")
    future = baskets[baskets["InvoiceNo"] == "104"].copy()
    future["InvoiceNo"], future["timestamp"], future["total_quantity"] = "999", STAMP("2012-06-01"), 1e6
    extended, _ = customer_feature_rows(ex, pd.concat([baskets, future]), "target_time")
    assert np.array_equal(base, extended)  # a basket far in the future changes nothing


def test_normalisers_are_fitted_on_train_only(processed, baskets):
    task = load_task(processed)
    a = build_feature_store(task, baskets, FEAT)
    changed = baskets.copy()
    later = changed["split"] != "train"
    changed.loc[later, "total_quantity"] = 1e5  # wreck every validation/test basket
    changed.loc[later, "total_value"] = 1e6
    b = build_feature_store(task, changed, FEAT)
    assert a.stats["basket"] == b.stats["basket"]  # train statistics do not move
    assert a.stats["customer"]["mean"][0] == b.stats["customer"]["mean"][0]
    assert a.customer_index == {"1": 1, "2": 2} and set(a.country_index) == {"UK", "DE"}


def test_customers_without_training_baskets_map_to_index_zero(processed, baskets):
    task = load_task(processed)
    assert build_feature_store(task, baskets, FEAT).example_arrays["validation"]["customer_idx"].tolist() == [
        1
    ]
    relabelled = baskets.copy()
    relabelled.loc[relabelled["CustomerID"] == "1", "split"] = (
        "validation"  # customer 1 now has no train basket
    )
    store = build_feature_store(task, relabelled, FEAT)
    assert store.customer_index == {"2": 1}
    assert store.example_arrays["validation"]["customer_idx"].tolist() == [0]  # unseen -> the shared id 0


@pytest.fixture
def data(processed):
    return build_training_data(processed, FEAT, GRAPH, quantity_samples=4)


def test_every_example_uses_a_snapshot_built_before_its_target(data):
    for split, ex in data.task.examples.items():
        for w, ts in zip(data.window[split], ex["target_timestamp"], strict=True):
            snap = data.snapshot_for(int(w))
            assert snap.cutoff_time <= ts, (split, snap.cutoff_time, ts)
            assert ts < snap.cutoff_time + pd.DateOffset(months=1)  # and it is the snapshot of its own month


def test_batches_cover_each_example_once_and_stay_in_one_window(data):
    for split in ("train", "validation", "test"):
        seen = []
        for w, idx in data.iter_batches(split, batch_size=2):
            assert (data.window[split][idx] == w).all()
            seen += idx.tolist()
        assert sorted(seen) == list(range(data.n_examples(split)))
    a = [i.tolist() for _, i in data.iter_batches("train", 2, np.random.default_rng(1))]
    b = [i.tolist() for _, i in data.iter_batches("train", 2, np.random.default_rng(1))]
    assert a == b  # same seed, same order


def test_collate_matches_the_examples(data):
    ex = data.task.examples["validation"]
    idx = np.arange(len(ex))
    batch, targets = data.collate("validation", idx)
    batch.validate(basket_feature_dim=8, numeric_dim=6)
    assert batch.basket_counts.tolist() == [len(h) for h in ex["history_invoices"]]
    assert (batch.item_offsets[1:] >= batch.item_offsets[:-1]).all()
    for row, i in enumerate(idx):
        assert targets.basket_hot[row].nonzero().flatten().tolist() == ex["target_idx"].iat[i].tolist()
    assert targets.history_hot.any() and (targets.quantity_mask.sum(1) > 0).all()
    k = targets.quantity_mask
    assert (targets.quantity_target[k] > 0).all() and (targets.quantity_target[~k] == 0).all()


def test_quantity_targets_are_log1p_of_the_real_quantity(data):
    _, targets = data.collate("validation", np.array([0]))  # basket 103: item 85123A, quantity 4
    assert targets.quantity_mask.sum() == 1
    assert targets.quantity_target[targets.quantity_mask].item() == pytest.approx(float(np.log1p(4)))
    assert data.task.items[int(targets.quantity_items[0, 0])] == "85123A"


def test_known_customers_keep_their_index_through_collate(data):
    assert data.store.example_arrays["test"]["customer_idx"].tolist() == [1]  # customer 1 has train baskets
    batch, _ = data.collate("test", np.array([0]))
    assert batch.customer_idx.tolist() == [1] and batch.country_idx.tolist() != [0]  # UK is a known country
