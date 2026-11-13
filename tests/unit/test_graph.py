import math

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from basketmind.graph import build_product_graph, calculate_ppmi, create_graph_snapshots

CFG = {"temporal_decay": {"enabled": False, "half_life_days": 90}, "min_cooccurrence": 1}
ITEMS = ["A", "B", "C", "D"]


def make_baskets():
    rows = [
        (["A", "B"], "2010-01-01"),
        (["A", "B"], "2010-01-02"),
        (["A", "C"], "2010-01-03"),
        (["C", "D"], "2010-03-01"),
        (["A", "B", "D"], "2010-04-01"),
    ]
    return pd.DataFrame({"items": [r[0] for r in rows], "timestamp": pd.to_datetime([r[1] for r in rows])})


def edges(g):
    return {
        (int(a), int(b)): float(w)
        for (a, b), w in zip(g.edge_index.T.tolist(), g.edge_weight.tolist(), strict=True)
    }


def test_ppmi_hand_computed():
    # 4 baskets: AB, AB, AC, B  -> c_A=3, c_B=3, c_C=1, c_AB=2, c_AC=1, N=4
    cooc = sp.csr_matrix(np.array([[3, 2, 1], [2, 3, 0], [1, 0, 1]], dtype=float))
    ppmi = calculate_ppmi(cooc, np.array([3.0, 3.0, 1.0]), 4.0).toarray()
    assert ppmi[0, 1] == 0.0  # log(2 * 4 / (3 * 3)) = log(8/9) < 0, clipped to 0
    assert ppmi[0, 2] == pytest.approx(math.log(1 * 4 / (3 * 1)))
    assert ppmi[1, 2] == 0.0 and np.diag(ppmi).sum() == 0.0  # no co-occurrence, no self loops
    assert (ppmi >= 0).all()


def test_min_cooccurrence_filters_rare_pairs():
    cooc = sp.csr_matrix(np.array([[2, 1], [1, 2]], dtype=float))
    assert calculate_ppmi(cooc, np.array([2.0, 2.0]), 2.0, min_cooccurrence=2).nnz == 0


def test_graph_is_symmetric_without_self_loops():
    g = build_product_graph(make_baskets(), pd.Timestamp("2011-01-01"), ITEMS, CFG)
    e = edges(g)
    assert all(a != b for a, b in e) and all(
        (b, a) in e and e[(b, a)] == pytest.approx(w) for (a, b), w in e.items()
    )
    assert g.n_nodes == 4 and g.n_baskets == 5 and (g.edge_weight > 0).all()


def test_future_baskets_cannot_change_a_snapshot():
    base = make_baskets()
    cutoff = pd.Timestamp("2010-02-01")
    clean = build_product_graph(base[base["timestamp"] < cutoff], cutoff, ITEMS, CFG)
    with_future = build_product_graph(base, cutoff, ITEMS, CFG)  # includes the March and April baskets
    assert edges(clean) == edges(with_future) and with_future.n_baskets == 3
    # a basket exactly at the cutoff is excluded too (strictly before)
    at_cutoff = pd.concat([base, pd.DataFrame({"items": [["B", "C"]], "timestamp": [cutoff]})])
    assert edges(build_product_graph(at_cutoff, cutoff, ITEMS, CFG)) == edges(clean)


def test_unknown_items_ignored_and_snapshots_per_boundary():
    b = make_baskets()
    b.at[0, "items"] = ["A", "B", "ZZZ"]  # not in the catalogue
    snaps = create_graph_snapshots(b, [pd.Timestamp("2010-02-01"), pd.Timestamp("2011-01-01")], ITEMS, CFG)
    assert list(snaps) == [pd.Timestamp("2010-02-01"), pd.Timestamp("2011-01-01")]
    assert (
        snaps[pd.Timestamp("2010-02-01")].n_baskets == 3 and snaps[pd.Timestamp("2011-01-01")].n_baskets == 5
    )


def test_temporal_decay_downweights_old_baskets():
    cutoff = pd.Timestamp("2010-06-01")
    flat = build_product_graph(make_baskets(), cutoff, ITEMS, CFG)
    cfg = {**CFG, "temporal_decay": {"enabled": True, "half_life_days": 30}}
    decayed = build_product_graph(make_baskets(), cutoff, ITEMS, cfg)
    # min_cooccurrence is judged on raw counts, so heavy decay must not delete edges
    assert decayed.edge_weight.numel() > 0 and decayed.n_baskets == flat.n_baskets
    assert edges(decayed) != edges(flat)  # but it does change the weights
