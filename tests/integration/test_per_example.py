import numpy as np
import pytest

from basketmind.baselines.models import ItemItemCF, Popularity
from basketmind.baselines.run import evaluate_model
from basketmind.evaluation.per_example import POLICIES, baseline_per_example, neural_per_example
from basketmind.training import engine
from basketmind.training.checkpoint import load_checkpoint
from basketmind.utils.config import load_config
from basketmind.utils.device import get_device

DEV = get_device()
KS = [1, 2]


def test_baseline_per_example_averages_to_the_standard_evaluation(training_data):
    task = training_data.task
    for model in (Popularity(), ItemItemCF()):
        model.fit(task, DEV, 0, lambda m: 0.0)
        per = baseline_per_example(model, task, "test", KS, DEV)
        for policy in POLICIES:
            standard = evaluate_model(model, task, "test", policy, KS, DEV)
            for metric in ("recall@1", "recall@2", "ndcg@2", "mrr"):
                got = per[policy][metric]
                expected = standard[metric]
                assert (np.isnan(expected) and len(got) == 0) or got.mean() == pytest.approx(expected), (
                    policy,
                    metric,
                )
            assert len(per[policy]["example_id"]) == standard["n_evaluated"]


def test_neural_per_example_matches_the_engines_validation_numbers(training_data, tmp_path):
    model_cfg, train_cfg = load_config("model"), load_config("training")
    train_cfg |= {"max_epochs": 1, "batch_size": 4, "selection_metric": "recall@1"}
    engine.run_experiment("v", training_data, model_cfg, train_cfg, 3, tmp_path, DEV, KS, False, False)
    net, _ = load_checkpoint(tmp_path / "best.pt", DEV)
    per = neural_per_example(net, training_data, "validation", KS, DEV)
    reference = engine.validate_one_epoch(net, training_data, "validation", train_cfg, model_cfg, KS, DEV)
    for policy in POLICIES:
        for metric in ("recall@1", "recall@2", "mrr"):
            expected = reference[f"{policy}.{metric}"]
            got = per[policy][metric]
            assert (np.isnan(expected) and len(got) == 0) or got.mean() == pytest.approx(
                expected, abs=1e-6
            ), metric


def test_example_ids_point_back_at_the_split_rows(training_data):
    task = training_data.task
    model = Popularity()
    model.fit(task, DEV, 0, lambda m: 0.0)
    ids = baseline_per_example(model, task, "test", KS, DEV)["full"]["example_id"]
    # every synthetic test example has a target, so ids are exactly the row positions
    assert ids.tolist() == list(range(len(task.examples["test"])))
