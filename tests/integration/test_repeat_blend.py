import pytest
import torch

from basketmind.evaluation import repeat_blend as rb
from basketmind.evaluation.per_example import POLICIES, neural_per_example
from basketmind.models import BasketMindModel
from basketmind.utils.config import load_config
from basketmind.utils.device import get_device
from basketmind.utils.seed import seed_everything

DEV = get_device()
KS = [1, 2]


@pytest.fixture
def net_and_data(training_data):
    task, store = training_data.task, training_data.store
    seed_everything(0)
    net = BasketMindModel(load_config("model"), task.n_items, store.n_customers, store.n_countries).to(DEV)
    return net, training_data


def test_zero_weight_is_the_unchanged_model(net_and_data):
    net, data = net_and_data
    cache = rb.build_cache(net, data, "test", (1.0,), DEV)
    base = neural_per_example(net, data, "test", KS, DEV)
    for policy in POLICIES:
        got = rb.evaluate(cache, 1.0, 0.0, KS, policy, keep=True).per_example()
        assert got["example_id"].tolist() == base[policy]["example_id"].tolist()
        for metric in ("recall@1", "recall@2", "mrr"):
            assert got[metric] == pytest.approx(base[policy][metric], abs=1e-6)


def test_the_bonus_never_changes_the_exclude_history_ranking(net_and_data):
    net, data = net_and_data
    cache = rb.build_cache(net, data, "test", (0.9,), DEV)
    for i, logits in enumerate(cache.logits):
        hist = cache.history[i]
        counts = cache.counts[0.9][i]
        assert ((counts > 0) == hist).all()  # the bonus is non-zero exactly on the history items
        plain = rb.blend(logits, counts, 0.0).masked_fill(hist, float("-inf"))
        strong = rb.blend(logits, counts, 64.0).masked_fill(hist, float("-inf"))
        assert torch.equal(plain, strong)  # so removing history leaves the unchanged model's scores


def test_a_large_weight_puts_the_customers_own_items_first():
    logits = torch.tensor([[5.0, 0.0, 0.0, 0.0]])  # the model likes item 0, which the customer never bought
    counts = torch.tensor([[0.0, 1.0, 3.0, 0.0]])  # they bought item 2 three times and item 1 once
    assert blend_rank(logits, counts, 0.0)[0] == 0
    assert blend_rank(logits, counts, 64.0)[:2].tolist() == [2, 1]


def blend_rank(logits, counts, lam):
    return rb.blend(logits, counts, lam).argsort(descending=True)[0]


def test_choose_returns_the_best_validation_cell(net_and_data):
    net, data = net_and_data
    cache = rb.build_cache(net, data, "validation", (1.0, 0.8), DEV)
    (decay, lam), grid = rb.choose(cache, KS, decays=(1.0, 0.8), lambdas=(0.0, 4.0), metric="recall@1")
    assert len(grid) == 4
    assert (decay, lam) in {(r["decay"], r["lam"]) for r in grid}
    assert next(r["val"] for r in grid if (r["decay"], r["lam"]) == (decay, lam)) == max(
        r["val"] for r in grid
    )
