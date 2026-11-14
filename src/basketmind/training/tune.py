"""Optuna search for the BasketMind training settings. VALIDATION only: the test split is never evaluated."""

import copy
import json
from pathlib import Path
from typing import Any

import optuna
import torch

from basketmind.baselines.tune import render_tuning_report
from basketmind.training.data import TrainingData, build_training_data
from basketmind.training.engine import run_experiment
from basketmind.utils.config import load_config, project_root
from basketmind.utils.device import get_device
from basketmind.utils.logging import get_logger

log = get_logger("training.tune")
DROPOUT_SITES = ("temporal_encoder", "graph_encoder", "customer_encoder", "fusion")


def suggest_neural(trial: optuna.Trial) -> dict[str, float]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),  # one value, applied to all four dropout sites
        "pos_weight": trial.suggest_float("pos_weight", 3.0, 100.0, log=True),
        "quantity_weight": trial.suggest_float("quantity_weight", 0.05, 1.0, log=True),
    }


def apply_params(
    params: dict[str, float], model_cfg: dict[str, Any], train_cfg: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copies of the configs with the sampled values applied."""
    model_cfg, train_cfg = copy.deepcopy(model_cfg), copy.deepcopy(train_cfg)
    train_cfg |= {k: params[k] for k in ("learning_rate", "weight_decay", "pos_weight")}
    for site in DROPOUT_SITES:
        model_cfg[site]["dropout"] = params["dropout"]
    model_cfg["loss_weights"]["quantity"] = params["quantity_weight"]
    return model_cfg, train_cfg


def tune_neural(
    data: TrainingData,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    device: torch.device,
    n_trials: int,
    out_dir: Path,
    storage: str | None = None,
    seed: int = 42,
    trial_epochs: int = 20,
    trial_patience: int = 5,
    ks: list[int] | None = None,
) -> optuna.Study:
    """Maximise validation Recall@10 over `n_trials` (resumable via `storage`).

    Trials are capped at `trial_epochs` and pruned (median rule) when clearly behind. The cap favours settings
    that learn quickly, so the winner should be retrained with the full budget before it is reported.
    """
    study = optuna.create_study(
        study_name="neural",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    done = len([t for t in study.trials if t.state.is_finished()])
    study.sampler = optuna.samplers.TPESampler(seed=seed + done)  # a fixed seed would replay draws on resume
    trial_cfg = train_cfg | {"max_epochs": trial_epochs, "early_stopping_patience": trial_patience}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_neural(trial)
        m_cfg, t_cfg = apply_params(params, model_cfg, trial_cfg)

        def report(epoch: int, score: float) -> None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        res = run_experiment(
            f"tune-{trial.number}",
            data,
            m_cfg,
            t_cfg,
            seed,
            out_dir / f"trial_{trial.number}",
            device,
            ks,
            evaluate_test=False,  # tuning never touches the test split
            log_mlflow=False,
            on_epoch=report,
        )
        trial.set_user_attr("best_epoch", res["best_epoch"])
        score = res["validation"]["full." + train_cfg["selection_metric"]]
        log.info("trial %d val=%.4f best_epoch=%d %s", trial.number, score, res["best_epoch"], params)
        return score

    if (remaining := n_trials - done) > 0:
        study.optimize(objective, n_trials=remaining)
    return study


def run_neural_tuning(n_trials: int = 24, root: Path | None = None) -> optuna.Study:
    root = root or project_root()
    train_cfg, model_cfg = load_config("training", root / "configs"), load_config("model", root / "configs")
    device = get_device(train_cfg["device"])
    out = root / "experiments" / "neural" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    data = build_training_data(
        root / "data" / "processed",
        load_config("features", root / "configs"),
        load_config("graph", root / "configs"),
        train_cfg["quantity_samples"],
    )
    study = tune_neural(
        data,
        model_cfg,
        train_cfg,
        device,
        n_trials,
        out,
        storage=f"sqlite:///{(out / 'optuna.db').as_posix()}",
    )
    baseline = None
    ref = root / "experiments" / "neural" / "lr3e-3" / "seed42" / "result.json"
    if ref.exists():
        baseline = json.loads(ref.read_text())["validation"]["full.recall@10"]
    study.trials_dataframe().to_csv(out / "neural_tuning_trials.csv", index=False)
    (root / "docs" / "neural_tuning.md").write_text(
        render_tuning_report("neural (BasketMind)", study, baseline, script="scripts/tune_neural.py"),
        encoding="utf-8",
    )
    return study
