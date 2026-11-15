"""Per-example results for every model, for confidence intervals, paired comparisons and slices.

This is ANALYSIS of finished models. Hyperparameters were fixed from validation before, and nothing here
feeds back into model selection. Baselines are not saved after training, so the stochastic ones are refitted
with the same configs and seeds (GPU runs are not bit-identical, so numbers can differ slightly from before).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from basketmind.baselines.models import Recommender
from basketmind.baselines.run import _models, evaluate_model
from basketmind.baselines.task import Task
from basketmind.evaluation.ranking import RankingMetrics
from basketmind.training.checkpoint import load_checkpoint
from basketmind.training.data import TrainingData, build_training_data
from basketmind.utils.config import load_config
from basketmind.utils.device import get_device, release_gpu
from basketmind.utils.logging import get_logger

log = get_logger("evaluation.per_example")
POLICIES = ("full", "exclude_history")
NEURAL = ("abl_graph_customer", "abl_full")  # experiment folders of the two BasketMind variants compared


def _frame(per: dict[str, dict[str, np.ndarray]], model: str, seed: int) -> pd.DataFrame:
    frames = []
    for policy, d in per.items():
        top = d.pop("top")
        f = pd.DataFrame(d).assign(model=model, seed=seed, policy=policy)
        f["top"] = list(top)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


@torch.no_grad()
def baseline_per_example(
    model: Recommender, task: Task, split: str, ks: list[int], device: torch.device, batch_size: int = 512
) -> dict[str, dict[str, np.ndarray]]:
    """Per-example metrics of a fitted baseline under both candidate policies."""
    meters = {p: RankingMetrics(ks, keep_examples=True) for p in POLICIES}
    ex = task.examples[split]
    for s in range(0, len(ex), batch_size):
        batch = ex.iloc[s : s + batch_size]
        scores, hot = model.score(task, batch, device), task.target_hot(batch, device)
        ids = np.arange(s, s + len(batch))
        meters["full"].update(scores, hot, ids=ids)
        meters["exclude_history"].update(scores, hot, task.history_hot(batch, device) > 0, ids=ids)
    return {p: m.per_example() for p, m in meters.items()}


@torch.no_grad()
def neural_per_example(
    model: torch.nn.Module,
    data: TrainingData,
    split: str,
    ks: list[int],
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-example metrics of a BasketMind model (eval mode) under both candidate policies."""
    model.eval()
    meters = {p: RankingMetrics(ks, keep_examples=True) for p in POLICIES}
    for window, idx in data.iter_batches(split, batch_size):
        batch, targets = data.collate(split, idx)
        batch, targets = batch.to(device), targets.to(device)
        logits = model(batch, data.snapshot_for(window).to(device))["basket_logits"]
        meters["full"].update(logits, targets.basket_hot, ids=idx)
        meters["exclude_history"].update(logits, targets.basket_hot, targets.history_hot, ids=idx)
    return {p: m.per_example() for p, m in meters.items()}


def collect_all(
    root: Path, split: str = "test", device: torch.device | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refit the baselines, score the BasketMind checkpoints; return per-example rows and a consistency log.

    The log compares each model's mean Recall@10 with the number reported earlier (baselines: results.csv, a
    refit, so small differences are expected; BasketMind: test_result.json, same checkpoint, so they match).
    """
    cfg, train_cfg = load_config("baselines", root / "configs"), load_config("training", root / "configs")
    ks = load_config("evaluation", root / "configs")["recommendation_k"]
    device = device or get_device(train_cfg["device"])
    data = build_training_data(
        root / "data" / "processed",
        load_config("features", root / "configs"),
        load_config("graph", root / "configs"),
        train_cfg["quantity_samples"],
    )
    task, frames, log_rows = data.task, [], []
    old = pd.read_csv(root / "experiments" / "baselines" / "results.csv")
    old = old[(old["split"] == split) & (old["policy"] == "full")].set_index(["model", "seed"])["recall@10"]

    for model in _models(cfg, train_cfg):
        for seed in train_cfg["seeds"] if model.stochastic else train_cfg["seeds"][:1]:
            model.fit(
                task,
                device,
                seed,
                lambda m: evaluate_model(m, task, "validation", "full", ks, device)["recall@10"],
            )
            per = baseline_per_example(model, task, split, ks, device)
            now = float(per["full"]["recall@10"].mean())
            log_rows.append(
                {"model": model.name, "seed": seed, "now": now, "before": float(old[(model.name, seed)])}
            )
            frames.append(_frame(per, model.name, seed))
            log.info("%s seed %d: mean recall@10 %.4f", model.name, seed, now)
            release_gpu()

    for name in NEURAL:
        for seed in train_cfg["seeds"]:
            run_dir = root / "experiments" / "neural" / name / f"seed{seed}"
            net, _ = load_checkpoint(run_dir / "best.pt", device)
            per = neural_per_example(net, data, split, ks, device)
            reported = pd.read_json(run_dir / "test_result.json", typ="series")["test"]["full.recall@10"]
            log_rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "now": float(per["full"]["recall@10"].mean()),
                    "before": float(reported),
                }
            )
            frames.append(_frame(per, name, seed))
            del net
            release_gpu()

    consistency = pd.DataFrame(log_rows).assign(diff=lambda d: d["now"] - d["before"])
    return pd.concat(frames, ignore_index=True), {"rows": consistency.to_dict("records")}
