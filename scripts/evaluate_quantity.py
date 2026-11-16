"""Score the quantity head and its baselines: python scripts/evaluate_quantity.py --split validation

Writes experiments/analysis/quantity_<split>.parquet: one row per target product with the real quantity, the
baseline predictions and one head prediction per seed. Nothing is tuned or selected here. The test split is
evaluated once: the script refuses to overwrite quantity_test.parquet. Then run scripts/analyse_quantity.py.
"""

import argparse

import pandas as pd

from basketmind.evaluation import quantity as q
from basketmind.training.checkpoint import load_checkpoint
from basketmind.training.data import build_training_data
from basketmind.utils.config import load_config, project_root
from basketmind.utils.device import get_device, release_gpu
from basketmind.utils.logging import get_logger

log = get_logger("quantity")
VARIANT = "abl_graph_customer"  # the selected BasketMind variant

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--device", default=None, help="e.g. cpu (default: the training config's device)")
    args = ap.parse_args()
    root = project_root()
    out = root / "experiments" / "analysis" / f"quantity_{args.split}.parquet"
    if args.split == "test" and out.exists():
        raise SystemExit(f"{out.name} exists: the test split is evaluated once")
    train_cfg = load_config("training", root / "configs")
    device = get_device(args.device or train_cfg["device"])
    proc = root / "data" / "processed"
    data = build_training_data(
        proc,
        load_config("features", root / "configs"),
        load_config("graph", root / "configs"),
        train_cfg["quantity_samples"],
    )
    baskets = pd.read_parquet(proc / "baskets.parquet", columns=["InvoiceNo", "split"])
    train_invoices = set(baskets.loc[baskets["split"] == "train", "InvoiceNo"])

    pairs = q.target_pairs(data, args.split)
    global_median, item_median = q.fit_baselines(data, train_invoices)
    frame = pd.concat([pairs, q.baseline_predictions(pairs, global_median, item_median)], axis=1)
    for seed in train_cfg["seeds"]:
        net, _ = load_checkpoint(
            root / "experiments" / "neural" / VARIANT / f"seed{seed}" / "best.pt", device
        )
        frame[f"head_seed{seed}"] = q.model_predictions(net, data, args.split, pairs, device)
        log.info("seed %d: head MAE %.3f", seed, (frame[f"head_seed{seed}"] - frame["actual"]).abs().mean())
        del net
        release_gpu()
    frame.to_parquet(out)
    item_mae = (frame["item_median"] - frame["actual"]).abs().mean()
    print(f"wrote {out.name}: {len(frame):,} pairs; item-median MAE {item_mae:.3f}")
