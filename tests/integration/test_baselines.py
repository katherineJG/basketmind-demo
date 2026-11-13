import math

import pandas as pd
import pytest
import torch

from basketmind.baselines.models import GRURecommender, ItemItemCF, MatrixFactorisation, Popularity
from basketmind.baselines.run import aggregate, evaluate_model
from basketmind.baselines.task import load_task
from basketmind.data.pipeline import run_m1
from basketmind.utils.config import load_config

DEVICE = torch.device("cpu")


@pytest.fixture
def task(synthetic_root):
    run_m1(root=synthetic_root)
    return load_task(synthetic_root / "data/processed")


def test_catalogue_comes_from_train_baskets_only(task):
    assert task.items == ["22000", "85123A"]  # synthetic train baskets hold only these products
    assert task.train_matrix.shape == (2, 2) and task.popularity.sum() > 0
    assert set(task.examples) == {"train", "validation", "test"}


def _models():
    cfg, train_cfg = load_config("baselines"), load_config("training")
    cfg["matrix_factorisation"] |= {"epochs": 20, "eval_every": 5, "patience": 2}
    train_cfg |= {"max_epochs": 3}
    return [
        Popularity(),
        ItemItemCF(),
        MatrixFactorisation(cfg["matrix_factorisation"]),
        GRURecommender(cfg["gru"], train_cfg),
    ]


@pytest.mark.parametrize("policy", ["full", "exclude_history"])
def test_every_baseline_trains_scores_and_evaluates(task, policy):
    for model in _models():
        model.fit(
            task,
            DEVICE,
            seed=0,
            evaluate=lambda m: evaluate_model(m, task, "validation", "full", [1, 2], DEVICE)["recall@1"],
        )
        batch = task.examples["test"]
        scores = model.score(task, batch, DEVICE)
        assert scores.shape == (len(batch), task.n_items) and torch.isfinite(scores).all(), model.name
        res = evaluate_model(model, task, "test", policy, [1, 2], DEVICE)
        assert all(0.0 <= res[k] <= 1.0 for k in ("recall@1", "recall@2", "mrr") if not math.isnan(res[k])), (
            model.name
        )


def test_aggregate_means_over_seeds():
    rows = [
        {"model": "m", "seed": s, "split": "test", "policy": "full", "recall@10": v, "n_evaluated": 5}
        for s, v in [(0, 0.2), (1, 0.4)]
    ]
    a = aggregate(pd.DataFrame(rows)).iloc[0]
    assert a["recall@10"] == pytest.approx(0.3) and a["recall@10_std"] == pytest.approx(0.1)
