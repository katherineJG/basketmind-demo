"""One-time test evaluation of finished runs, using each run's best-validation checkpoint."""

import json
from pathlib import Path
from typing import Any

import torch

from basketmind.training.checkpoint import load_checkpoint
from basketmind.training.data import TrainingData, build_training_data
from basketmind.training.engine import validate_one_epoch
from basketmind.utils.config import load_config
from basketmind.utils.device import get_device, release_gpu


def run_dirs(root: Path, runs: list[tuple[str, int]]) -> list[Path]:
    return [root / "experiments" / "neural" / name / f"seed{seed}" for name, seed in runs]


def check_ready(dirs: list[Path]) -> None:
    """Refuse to continue if any run was already evaluated on test, or is not finished."""
    done = [str(d) for d in dirs if (d / "test_result.json").exists()]
    if done:
        raise FileExistsError(f"test was already evaluated for: {done}. Nothing was run.")
    missing = [str(d) for d in dirs if not (d / "best.pt").exists() or not (d / "result.json").exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint or result for: {missing}")


def evaluate_test_once(
    root: Path,
    runs: list[tuple[str, int]],
    data: TrainingData | None = None,
    ks: list[int] | None = None,
    device: torch.device | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate `runs` (name, seed) on the test split and write test_result.json next to each checkpoint.

    All runs are checked before any is evaluated, and an existing test_result.json is never overwritten, so a
    test score cannot be recomputed or quietly re-chosen after it has been seen.
    """
    dirs = run_dirs(root, runs)
    check_ready(dirs)
    train_cfg = load_config("training", root / "configs")
    device = device or get_device(train_cfg["device"])
    ks = ks or load_config("evaluation", root / "configs")["recommendation_k"]
    data = data or build_training_data(
        root / "data" / "processed",
        load_config("features", root / "configs"),
        load_config("graph", root / "configs"),
        train_cfg["quantity_samples"],
    )
    results = {}
    for (name, seed), d in zip(runs, dirs, strict=True):
        run_cfg = json.loads((d / "result.json").read_text())["train_config"]  # this run's own settings
        model, payload = load_checkpoint(d / "best.pt", device)
        test = validate_one_epoch(model, data, "test", run_cfg, model.init_args["cfg"], ks, device)
        out = {
            "name": name,
            "seed": int(seed),
            "checkpoint_epoch": payload["metadata"]["epoch"],
            "checkpoint_val_score": payload["metadata"]["val_score"],
            "test": test,
        }
        (d / "test_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        results[f"{name}:{seed}"] = out
        del model
        release_gpu()
    return results
