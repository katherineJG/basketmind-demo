"""Train BasketMind on the processed data: `python scripts/train.py --name full --seed 42`.

Selection, early stopping and checkpointing use validation only; the test split is evaluated once at the end.
Use --no-test for validation-only runs (tuning, smoke tests).
"""

import argparse
import copy
import time

import mlflow
import optuna

from basketmind.training.data import build_training_data
from basketmind.training.engine import run_experiment
from basketmind.training.tune import apply_params
from basketmind.utils.config import load_config, project_root
from basketmind.utils.device import get_device, release_gpu

BRANCHES = ("temporal", "graph", "customer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="full", help="experiment name (also the output folder)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--branches", default=",".join(BRANCHES), help="comma list of enabled branches")
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--pos-weight", type=float)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--tuned", action="store_true", help="apply the best settings of the Optuna search")
    args = ap.parse_args()

    root = project_root()
    model_cfg, train_cfg = load_config("model"), load_config("training")
    if args.tuned:
        storage = f"sqlite:///{(root / 'experiments' / 'neural' / 'tuning' / 'optuna.db').as_posix()}"
        best = optuna.load_study(study_name="neural", storage=storage).best_params
        model_cfg, train_cfg = apply_params(best, model_cfg, train_cfg)
        print("tuned settings:", {k: round(v, 5) for k, v in best.items()}, flush=True)
    enabled = set(args.branches.split(","))
    model_cfg["branches"] = {b: b in enabled for b in BRANCHES}
    if args.max_epochs:
        train_cfg["max_epochs"] = args.max_epochs
    if args.pos_weight:
        train_cfg["pos_weight"] = args.pos_weight
    if args.lr:
        train_cfg["learning_rate"] = args.lr

    mlflow.set_tracking_uri(f"sqlite:///{(root / 'experiments' / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("neural")
    t0 = time.time()
    data = build_training_data(
        root / "data" / "processed",
        load_config("features"),
        load_config("graph"),
        train_cfg["quantity_samples"],
    )
    print(f"data ready in {time.time() - t0:.0f}s", flush=True)
    out_dir = root / "experiments" / "neural" / args.name / f"seed{args.seed}"
    res = run_experiment(
        args.name,
        data,
        copy.deepcopy(model_cfg),
        train_cfg,
        args.seed,
        out_dir,
        get_device(train_cfg["device"]),
        evaluate_test=not args.no_test,
    )
    keys = ["full.recall@10", "full.ndcg@10", "full.mrr", "exclude_history.recall@10"]
    summary = f"{res['fit_seconds']}s, {res['parameters']:,} params"
    print(f"best epoch {res['best_epoch']} of {res['epochs_run']}, {summary}")
    for split in ("validation", "test"):
        if split in res:
            print(split, {k: round(res[split][k], 4) for k in keys}, flush=True)
    release_gpu()


if __name__ == "__main__":
    main()
