import json

import numpy as np
import pytest
import torch

from basketmind.models import BasketMindModel
from basketmind.training import engine
from basketmind.training.checkpoint import load_checkpoint
from basketmind.utils.config import load_config
from basketmind.utils.device import get_device

DEV = get_device()
KS = [1, 2]


def cfgs(**train):
    model_cfg, train_cfg = load_config("model"), load_config("training")
    train_cfg |= {
        "max_epochs": 3,
        "batch_size": 4,
        "early_stopping_patience": 7,
        "selection_metric": "recall@1",  # the tests rank with k in [1, 2]
    } | train
    return model_cfg, train_cfg


def test_one_epoch_trains_and_validation_reports_both_policies(training_data):
    model_cfg, train_cfg = cfgs()
    task, store = training_data.task, training_data.store
    model = BasketMindModel(model_cfg, task.n_items, store.n_customers, store.n_countries).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = engine.train_one_epoch(
        model, training_data, opt, train_cfg, model_cfg, np.random.default_rng(0), DEV
    )
    assert set(losses) == {"total", "basket", "quantity"} and all(np.isfinite(v) for v in losses.values())
    val = engine.validate_one_epoch(model, training_data, "validation", train_cfg, model_cfg, KS, DEV)
    for key in ("loss_total", "full.recall@1", "full.mrr", "exclude_history.recall@2", "full.n_evaluated"):
        assert key in val


def test_training_reduces_the_training_loss(training_data):
    model_cfg, train_cfg = cfgs()
    task, store = training_data.task, training_data.store
    model = BasketMindModel(model_cfg, task.n_items, store.n_customers, store.n_countries).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)
    first = engine.train_one_epoch(model, training_data, opt, train_cfg, model_cfg, rng, DEV)["total"]
    for _ in range(15):
        last = engine.train_one_epoch(model, training_data, opt, train_cfg, model_cfg, rng, DEV)["total"]
    assert last < first


def test_run_experiment_writes_checkpoint_and_result_and_touches_test_once_at_the_end(
    training_data, tmp_path, monkeypatch
):
    calls = []
    real = engine.validate_one_epoch

    def spy(model, data, split, *a, **k):
        calls.append(split)
        return real(model, data, split, *a, **k)

    monkeypatch.setattr(engine, "validate_one_epoch", spy)
    model_cfg, train_cfg = cfgs()
    res = engine.run_experiment(
        "full", training_data, model_cfg, train_cfg, 7, tmp_path, DEV, KS, log_mlflow=False
    )

    assert calls.count("test") == 1 and calls[-1] == "test"  # once, after all training epochs
    assert calls[:-2] == ["validation"] * res["epochs_run"]  # during training only validation is used
    assert (tmp_path / "best.pt").exists() and (tmp_path / "result.json").exists()
    on_disk = json.loads((tmp_path / "result.json").read_text())
    assert on_disk["seed"] == 7 and on_disk["best_epoch"] == res["best_epoch"] and "test" in on_disk
    assert res["parameters"] > 0 and len(res["history"]) == res["epochs_run"]


def test_validation_only_runs_never_touch_test(training_data, tmp_path):
    model_cfg, train_cfg = cfgs(max_epochs=1)
    res = engine.run_experiment(
        "tuning",
        training_data,
        model_cfg,
        train_cfg,
        1,
        tmp_path,
        DEV,
        KS,
        evaluate_test=False,
        log_mlflow=False,
    )
    assert "test" not in res and "validation" in res


def test_early_stopping_and_best_checkpoint_selection(training_data, tmp_path, monkeypatch):
    scores = iter([0.1, 0.3, 0.2, 0.2, 0.25, 0.9, 0.9])  # best at epoch 2, then 2 epochs without improvement

    def scripted(model, data, split, *a, **k):
        return {"full.recall@10": next(scores, 0.0), "loss_total": 0.0}

    monkeypatch.setattr(engine, "validate_one_epoch", scripted)
    model_cfg, train_cfg = cfgs(max_epochs=20, early_stopping_patience=2)
    train_cfg["selection_metric"] = "recall@10"
    res = engine.run_experiment(
        "es", training_data, model_cfg, train_cfg, 0, tmp_path, DEV, KS, log_mlflow=False
    )
    assert res["best_epoch"] == 2 and res["epochs_run"] == 4  # epochs 3 and 4 did not beat 0.3 -> stop
    _, payload = load_checkpoint(tmp_path / "best.pt", DEV)
    assert payload["metadata"]["epoch"] == 2 and payload["metadata"]["val_score"] == 0.3


def test_unavailable_selection_metric_fails_with_a_clear_error(training_data, tmp_path):
    model_cfg, train_cfg = cfgs(selection_metric="recall@10")  # not computed when ks = [1, 2]
    with pytest.raises(KeyError, match="was not computed"):
        engine.run_experiment(
            "x", training_data, model_cfg, train_cfg, 0, tmp_path, DEV, KS, log_mlflow=False
        )


def test_same_seed_gives_the_same_batch_order(training_data):
    a = [i.tolist() for _, i in training_data.iter_batches("train", 4, np.random.default_rng(5))]
    b = [i.tolist() for _, i in training_data.iter_batches("train", 4, np.random.default_rng(5))]
    assert a == b
