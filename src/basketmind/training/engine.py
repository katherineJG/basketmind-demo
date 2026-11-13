"""Training engine: one epoch of training, validation, and a full seeded experiment."""

import copy
import json
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch

from basketmind.evaluation.ranking import RankingMetrics
from basketmind.models import BasketMindModel
from basketmind.training.checkpoint import load_checkpoint, save_checkpoint
from basketmind.training.data import TrainingData
from basketmind.training.losses import loss_settings, multitask_loss
from basketmind.utils.config import load_config
from basketmind.utils.device import describe_device, release_gpu
from basketmind.utils.logging import get_logger
from basketmind.utils.seed import seed_everything

log = get_logger("training.engine")
POLICIES = ("full", "exclude_history")


def train_one_epoch(
    model: BasketMindModel,
    data: TrainingData,
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, float]:
    """One pass over the training examples (shuffled, grouped by graph snapshot). Returns mean losses."""
    model.train()
    settings = loss_settings(train_cfg, model_cfg)
    sums = {"total": 0.0, "basket": 0.0, "quantity": 0.0}
    n = 0
    for window, idx in data.iter_batches("train", train_cfg["batch_size"], rng):
        batch, targets = data.collate("train", idx, rng)
        batch, targets = batch.to(device), targets.to(device)
        out = model(batch, data.snapshot_for(window).to(device), targets.quantity_items)
        losses = multitask_loss(out, targets, **settings)
        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
        optimizer.step()
        for k in sums:
            sums[k] += losses[k].item() * len(idx)
        n += len(idx)
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def validate_one_epoch(
    model: BasketMindModel,
    data: TrainingData,
    split: str,
    train_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    ks: list[int],
    device: torch.device,
) -> dict[str, float]:
    """Loss and ranking metrics (both candidate policies) over `split`, in fixed order, no shuffling."""
    model.eval()
    settings = loss_settings(train_cfg, model_cfg)
    meters = {p: RankingMetrics(ks) for p in POLICIES}
    sums = {"total": 0.0, "basket": 0.0, "quantity": 0.0}
    n = 0
    for window, idx in data.iter_batches(split, train_cfg["batch_size"]):
        batch, targets = data.collate(split, idx)
        batch, targets = batch.to(device), targets.to(device)
        out = model(batch, data.snapshot_for(window).to(device), targets.quantity_items)
        losses = multitask_loss(out, targets, **settings)
        for k in sums:
            sums[k] += losses[k].item() * len(idx)
        n += len(idx)
        meters["full"].update(out["basket_logits"], targets.basket_hot)
        meters["exclude_history"].update(out["basket_logits"], targets.basket_hot, targets.history_hot)
    result = {f"loss_{k}": v / n for k, v in sums.items()}
    for policy, meter in meters.items():
        result |= {f"{policy}.{k}": v for k, v in meter.compute().items()}
    return result


def run_experiment(
    name: str,
    data: TrainingData,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    seed: int,
    out_dir: Path,
    device: torch.device,
    ks: list[int] | None = None,
    evaluate_test: bool = True,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Train with early stopping on validation `selection_metric`; keep the best checkpoint; test once.

    The test split is touched exactly once, after training, with the best-validation checkpoint. Nothing
    (stopping, scheduler, checkpoint choice, tuning) ever sees it.
    """
    ks = ks or load_config("evaluation")["recommendation_k"]
    sel_key = "full." + train_cfg["selection_metric"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    task, store = data.task, data.store
    model = BasketMindModel(copy.deepcopy(model_cfg), task.n_items, store.n_customers, store.n_countries).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=train_cfg["scheduler_factor"], patience=train_cfg["scheduler_patience"]
    )
    run = mlflow.start_run(run_name=f"{name}-seed{seed}") if log_mlflow else None
    try:
        if run:
            mlflow.log_params(
                {
                    "model": name,
                    "seed": seed,
                    "device": str(device),
                    **{f"train.{k}": v for k, v in train_cfg.items() if not isinstance(v, (list, dict))},
                }
            )
            mlflow.log_params({f"branch.{k}": v for k, v in model_cfg["branches"].items()})
        best, best_epoch, bad, history = -1.0, 0, 0, []
        start = time.time()
        ckpt = out_dir / "best.pt"
        for epoch in range(1, train_cfg["max_epochs"] + 1):
            train_losses = train_one_epoch(model, data, optimizer, train_cfg, model_cfg, rng, device)
            val = validate_one_epoch(model, data, "validation", train_cfg, model_cfg, ks, device)
            if sel_key not in val:
                raise KeyError(f"selection metric {sel_key!r} was not computed; available: {sorted(val)}")
            score = val[sel_key]
            scheduler.step(score)
            history.append(
                {"epoch": epoch, "train": train_losses, "val": val, "lr": optimizer.param_groups[0]["lr"]}
            )
            if run:
                mlflow.log_metrics({f"train_{k}": v for k, v in train_losses.items()}, step=epoch)
                mlflow.log_metrics({f"val_{k.replace('@', '_at_')}": v for k, v in val.items()}, step=epoch)
            log.info(
                "%s seed %d epoch %d train %.4f val %s=%.4f",
                name,
                seed,
                epoch,
                train_losses["total"],
                sel_key,
                score,
            )
            if score > best:
                best, best_epoch, bad = score, epoch, 0
                save_checkpoint(
                    ckpt, model, {"epoch": epoch, "val_score": float(score), "seed": seed}, overwrite=True
                )
            else:
                bad += 1
                if bad >= train_cfg["early_stopping_patience"]:
                    break
        fit_seconds = time.time() - start
        model, _ = load_checkpoint(ckpt, device)  # the best-validation weights
        result: dict[str, Any] = {
            "name": name,
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "fit_seconds": round(fit_seconds, 1),
            "parameters": sum(p.numel() for p in model.parameters()),
            "validation": validate_one_epoch(model, data, "validation", train_cfg, model_cfg, ks, device),
            "model_config": model_cfg,
            "train_config": {k: v for k, v in train_cfg.items()},
            "device": describe_device(),
            "history": history,
        }
        if evaluate_test:
            result["test"] = validate_one_epoch(model, data, "test", train_cfg, model_cfg, ks, device)
        if run:
            mlflow.log_metrics(
                {f"final_val_{k.replace('@', '_at_')}": v for k, v in result["validation"].items()}
            )
            if evaluate_test:
                mlflow.log_metrics({f"test_{k.replace('@', '_at_')}": v for k, v in result["test"].items()})
        (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result
    finally:
        if run:
            mlflow.end_run()
        model = optimizer = scheduler = None  # drop the GPU tensors before releasing (see release_gpu)
        release_gpu()
