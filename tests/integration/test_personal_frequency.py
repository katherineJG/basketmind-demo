import numpy as np
import pandas as pd
import pytest
import torch

from basketmind.baselines.models import PersonalFrequency
from basketmind.utils.device import get_device

DEV = get_device()


def two_invoices(task):
    """Two real (synthetic-data) invoices as a history, plus their item index arrays."""
    inv = [i for i, items in task.basket_items.items() if len(items)][:2]
    return inv, [task.basket_items[i] for i in inv]


def test_history_weighted_counts_recent_baskets_more(training_data):
    task = training_data.task
    inv, items = two_invoices(task)
    batch = pd.DataFrame({"history_invoices": [inv]})
    expected = np.zeros(task.n_items, dtype=np.float32)
    expected[items[0]] += 0.5  # older basket: one step back
    expected[items[1]] += 1.0  # newest basket
    got = task.history_weighted(batch, DEV, decay=0.5)[0].cpu().numpy()
    assert got == pytest.approx(expected)
    plain = task.history_weighted(batch, DEV)[0].cpu().numpy()  # decay 1 = plain counts
    assert plain.sum() == pytest.approx(len(items[0]) + len(items[1]))


def test_fit_picks_the_best_validation_setting_and_scores_own_items_first(training_data):
    task = training_data.task
    model = PersonalFrequency("personal_frequency_cf", decays=(1.0, 0.8), cf_weights=(0.0, 1.0))
    seen = []

    def evaluate(m):  # a fake validation score that prefers decay 0.8 with the CF blend
        seen.append((m.decay, m.w))
        return (m.decay == 0.8) + 0.5 * (m.w == 1.0)

    info = model.fit(task, DEV, 0, evaluate)
    assert (model.decay, model.w) == (0.8, 1.0)
    assert len(seen) == len(info["grid"]) == 4
    inv, items = two_invoices(task)
    scores = model.score(task, pd.DataFrame({"history_invoices": [inv]}), DEV)
    assert scores.shape == (1, task.n_items)
    assert torch.isfinite(scores).all()
    bought = np.unique(np.concatenate(items))
    assert scores[0, bought].min() >= scores[0].median()  # bought items outrank the typical item


def test_purely_personal_model_ignores_the_item_similarity(training_data):
    task = training_data.task
    model = PersonalFrequency("personal_frequency", decays=(1.0,), cf_weights=(0.0,))
    model.fit(task, DEV, 0, lambda m: 0.0)
    assert not hasattr(model, "sim")
    inv, _ = two_invoices(task)
    assert model.score(task, pd.DataFrame({"history_invoices": [inv]}), DEV).shape == (1, task.n_items)
