"""BasketMind plus a repeat-purchase score: choose (decay, weight) per seed on VALIDATION, test once.

Reuses the three saved checkpoints of the selected variant (abl_graph_customer); nothing is retrained.
Writes experiments/analysis/per_example_repeat_blend.parquet and repeat_blend.json. Refuses to run twice
(test is evaluated once per model). Then run scripts/analyse_repeats.py.
"""

import json
import sys

import pandas as pd

from basketmind.evaluation import repeat_blend as rb
from basketmind.evaluation.per_example import _frame
from basketmind.training.checkpoint import load_checkpoint
from basketmind.training.data import build_training_data
from basketmind.utils.config import load_config, project_root
from basketmind.utils.device import get_device, release_gpu
from basketmind.utils.logging import get_logger

log = get_logger("repeat_blend")
VARIANT, NAME = "abl_graph_customer", "abl_graph_customer_repeat"

if __name__ == "__main__":
    root = project_root()
    out = root / "experiments" / "analysis"
    target = out / "per_example_repeat_blend.parquet"
    if target.exists():
        raise SystemExit(f"{target.name} exists: this model was already evaluated on test (once per model)")
    cfg, train_cfg = load_config("features", root / "configs"), load_config("training", root / "configs")
    ks = load_config("evaluation", root / "configs")["recommendation_k"]
    device = get_device(sys.argv[1] if len(sys.argv) > 1 else train_cfg["device"])  # e.g. `cpu`
    data = build_training_data(
        root / "data" / "processed",
        cfg,
        load_config("graph", root / "configs"),
        train_cfg["quantity_samples"],
    )
    frames, summary = [], {}
    for seed in train_cfg["seeds"]:
        run_dir = root / "experiments" / "neural" / VARIANT / f"seed{seed}"
        net, _ = load_checkpoint(run_dir / "best.pt", device)
        (decay, lam), grid = rb.choose(rb.build_cache(net, data, "validation", rb.DECAYS, device), ks)
        base_val = next(r["val"] for r in grid if r["lam"] == 0.0)
        best_val = next(r["val"] for r in grid if (r["decay"], r["lam"]) == (decay, lam))
        log.info(
            "seed %d: decay %.1f, weight %.1f, validation recall@10 %.4f (unchanged %.4f)",
            seed,
            decay,
            lam,
            best_val,
            base_val,
        )
        test = rb.build_cache(net, data, "test", (decay,), device)  # the single test evaluation of this model
        per = {p: rb.evaluate(test, decay, lam, ks, p, keep=True).per_example() for p in rb.POLICIES}
        reported = json.loads((run_dir / "test_result.json").read_text())["test"]
        gap = abs(float(per["exclude_history"]["recall@10"].mean()) - reported["exclude_history.recall@10"])
        if gap > 1e-4:  # the bonus is zero for non-history items, so this policy must be unchanged
            raise SystemExit(
                f"seed {seed}: exclude_history recall@10 differs from the reported value by {gap:.2g}"
            )
        summary[seed] = {
            "decay": decay,
            "lam": lam,
            "validation_recall10": best_val,
            "validation_recall10_unchanged": base_val,
            "grid": grid,
            "test_recall10_full": float(per["full"]["recall@10"].mean()),
            "test_recall10_full_reported_unchanged": reported["full.recall@10"],
            "test_recall10_exclude_history": float(per["exclude_history"]["recall@10"].mean()),
        }
        frames.append(_frame(per, NAME, seed))
        del net, test
        release_gpu()
    pd.concat(frames, ignore_index=True).to_parquet(target)
    (out / "repeat_blend.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    for seed, s in summary.items():
        print(
            seed,
            "decay",
            s["decay"],
            "weight",
            s["lam"],
            "val",
            round(s["validation_recall10"], 4),
            "test full",
            round(s["test_recall10_full"], 4),
            "(unchanged",
            round(s["test_recall10_full_reported_unchanged"], 4),
            ")",
        )
