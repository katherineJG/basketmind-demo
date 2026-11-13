import pytest
import torch

from basketmind.baselines import tune
from basketmind.baselines.task import load_task
from basketmind.data.pipeline import run_m1
from basketmind.utils.config import load_config

CPU = torch.device("cpu")


@pytest.fixture
def task(synthetic_root):
    run_m1(root=synthetic_root)
    return load_task(synthetic_root / "data/processed")


@pytest.fixture
def train_cfg():
    return load_config("training") | {"max_epochs": 2}


@pytest.mark.parametrize("name", sorted(tune.SPACES))
def test_search_runs_and_only_evaluates_validation(name, task, train_cfg, monkeypatch):
    seen_splits = []
    real_eval = tune.evaluate_model

    def spy(model, task, split, *a, **k):
        seen_splits.append(split)
        return real_eval(model, task, split, *a, **k)

    monkeypatch.setattr(tune, "evaluate_model", spy)
    study = tune.tune_model(name, task, train_cfg, [1, 2], CPU, n_trials=2, selection_metric="recall@1")

    assert len(study.trials) == 2 and all(t.state.is_finished() for t in study.trials)
    assert seen_splits and set(seen_splits) == {"validation"}  # test is never touched during tuning
    assert "val recall@10" in tune.render_tuning_report(name, study, None)


def test_resume_does_not_rerun_finished_trials(task, train_cfg, tmp_path):
    storage = f"sqlite:///{(tmp_path / 'o.db').as_posix()}"
    args = ("gru", task, train_cfg, [1, 2], CPU)
    tune.tune_model(*args, n_trials=1, storage=storage, selection_metric="recall@1")
    study = tune.tune_model(*args, n_trials=2, storage=storage, selection_metric="recall@1")
    assert len(study.trials) == 2
    assert study.trials[0].params != study.trials[1].params  # resume must not replay the same draws


def test_studies_for_different_models_do_not_mix(task, train_cfg, tmp_path):
    storage = f"sqlite:///{(tmp_path / 'o.db').as_posix()}"
    for name in ("gru", "matrix_factorisation"):
        s = tune.tune_model(
            name, task, train_cfg, [1, 2], CPU, n_trials=1, storage=storage, selection_metric="recall@1"
        )
        assert len(s.trials) == 1
    assert "dim" in s.trials[0].params and "hidden_dim" not in s.trials[0].params
