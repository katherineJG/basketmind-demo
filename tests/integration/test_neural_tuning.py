import optuna
import pytest
import torch

from basketmind.training import engine
from basketmind.training import tune as ntune
from basketmind.utils.config import load_config
from basketmind.utils.device import get_device

DEV = get_device()
PARAMS = {
    "learning_rate": 0.004,
    "weight_decay": 1e-5,
    "dropout": 0.3,
    "pos_weight": 40.0,
    "quantity_weight": 0.2,
}


def cfgs():
    train_cfg = load_config("training") | {"batch_size": 4, "selection_metric": "recall@1"}
    return load_config("model"), train_cfg


def test_apply_params_sets_every_site_and_leaves_the_originals_alone():
    model_cfg, train_cfg = cfgs()
    before = (repr(model_cfg), repr(train_cfg))
    m, t = ntune.apply_params(PARAMS, model_cfg, train_cfg)
    assert (t["learning_rate"], t["weight_decay"], t["pos_weight"]) == (0.004, 1e-5, 40.0)
    assert all(m[site]["dropout"] == 0.3 for site in ntune.DROPOUT_SITES)
    assert m["loss_weights"]["quantity"] == 0.2 and m["loss_weights"]["basket"] == 1.0
    assert (repr(model_cfg), repr(train_cfg)) == before  # inputs are untouched (deep copies)


def test_sampled_values_stay_inside_the_documented_ranges():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    seen = []
    study.optimize(lambda tr: seen.append(ntune.suggest_neural(tr)) or 0.0, n_trials=60)
    ranges = {
        "learning_rate": (1e-3, 1e-2),
        "weight_decay": (1e-6, 1e-2),
        "dropout": (0.0, 0.5),
        "pos_weight": (3.0, 100.0),
        "quantity_weight": (0.05, 1.0),
    }
    for params in seen:
        assert set(params) == set(ranges)
        assert all(lo <= params[k] <= hi for k, (lo, hi) in ranges.items())


def test_search_only_ever_evaluates_validation(training_data, tmp_path, monkeypatch):
    splits = []
    real = engine.validate_one_epoch

    def spy(model, data, split, *a, **k):
        splits.append(split)
        return real(model, data, split, *a, **k)

    monkeypatch.setattr(engine, "validate_one_epoch", spy)
    model_cfg, train_cfg = cfgs()
    study = ntune.tune_neural(
        training_data, model_cfg, train_cfg, DEV, n_trials=2, out_dir=tmp_path, trial_epochs=1, ks=[1, 2]
    )
    assert len(study.trials) == 2 and all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert splits and set(splits) == {"validation"}  # the test split is never touched while tuning
    assert (tmp_path / "trial_0" / "best.pt").exists() and "test" not in str(splits)
    assert all("best_epoch" in t.user_attrs for t in study.trials)


def test_poor_trials_are_pruned_and_the_search_carries_on(training_data, tmp_path, monkeypatch):
    monkeypatch.setattr(optuna.trial.Trial, "should_prune", lambda self: True)
    model_cfg, train_cfg = cfgs()
    study = ntune.tune_neural(
        training_data, model_cfg, train_cfg, DEV, n_trials=2, out_dir=tmp_path, trial_epochs=3, ks=[1, 2]
    )
    assert [t.state for t in study.trials] == [optuna.trial.TrialState.PRUNED] * 2
    assert all(
        t.value is not None for t in study.trials
    )  # Optuna keeps the last score; the report must ignore it
    assert "Top 10 trials" in ntune.render_tuning_report("neural", study, None)  # no crash without values


def test_resume_continues_and_never_replays_draws(training_data, tmp_path):
    model_cfg, train_cfg = cfgs()
    storage = f"sqlite:///{(tmp_path / 'o.db').as_posix()}"
    kw = dict(out_dir=tmp_path, trial_epochs=1, ks=[1, 2], storage=storage)
    ntune.tune_neural(training_data, model_cfg, train_cfg, DEV, n_trials=1, **kw)
    study = ntune.tune_neural(training_data, model_cfg, train_cfg, DEV, n_trials=2, **kw)
    assert len(study.trials) == 2 and study.trials[0].params != study.trials[1].params


def test_engine_callback_can_abort_a_run(training_data, tmp_path):
    model_cfg, train_cfg = cfgs()

    class Stop(Exception):
        pass

    def stop(epoch, score):
        raise Stop

    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    with pytest.raises(Stop):
        engine.run_experiment(
            "x",
            training_data,
            model_cfg,
            train_cfg,
            0,
            tmp_path,
            DEV,
            [1, 2],
            log_mlflow=False,
            on_epoch=stop,
        )
    if torch.cuda.is_available():  # an aborted run must not leave its model and optimiser on the GPU
        leaked = torch.cuda.memory_allocated() - before
        assert leaked < 1 * 2**20, f"{leaked / 2**20:.1f} MiB still allocated after an aborted run"
