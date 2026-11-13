import torch

from basketmind.baselines import tune
from basketmind.baselines.task import load_task
from basketmind.data.pipeline import run_m1
from basketmind.utils.config import load_config


def test_search_runs_and_only_evaluates_validation(synthetic_root, monkeypatch):
    run_m1(root=synthetic_root)
    task = load_task(synthetic_root / "data/processed")
    seen_splits = []
    real_eval = tune.evaluate_model

    def spy(model, task, split, *a, **k):
        seen_splits.append(split)
        return real_eval(model, task, split, *a, **k)

    monkeypatch.setattr(tune, "evaluate_model", spy)
    train_cfg = load_config("training") | {"max_epochs": 2}
    study = tune.tune_gru(
        task, train_cfg, [1, 2], torch.device("cpu"), n_trials=2, selection_metric="recall@1"
    )

    assert len(study.trials) == 2 and all(t.state.is_finished() for t in study.trials)
    assert seen_splits and set(seen_splits) == {"validation"}  # test is never touched during tuning
    assert "val recall@10" in tune.render_tuning_report(study, None)


def test_resume_does_not_rerun_finished_trials(synthetic_root, tmp_path):
    run_m1(root=synthetic_root)
    task = load_task(synthetic_root / "data/processed")
    train_cfg = load_config("training") | {"max_epochs": 1}
    storage = f"sqlite:///{(tmp_path / 'o.db').as_posix()}"
    args = (task, train_cfg, [1, 2], torch.device("cpu"))
    tune.tune_gru(*args, n_trials=1, storage=storage, selection_metric="recall@1")
    study = tune.tune_gru(*args, n_trials=2, storage=storage, selection_metric="recall@1")
    assert len(study.trials) == 2
    assert study.trials[0].params != study.trials[1].params  # resume must not replay the same draws
