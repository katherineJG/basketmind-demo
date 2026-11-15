"""Copy the selected BasketMind checkpoint to artifacts/ for the API, with a metadata file describing it.

    python scripts/export_model.py                       # best-validation seed of abl_graph_customer
    python scripts/export_model.py --variant abl_full --force

The seed is chosen by validation Recall@10 only. The test numbers in the metadata are copied from each run's
existing test_result.json (written once by evaluate_test.py); nothing is evaluated here.
"""

import argparse
import json
import shutil

from basketmind.utils.config import project_root

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="abl_graph_customer")
    ap.add_argument("--force", action="store_true", help="overwrite an existing exported checkpoint")
    args = ap.parse_args()
    root = project_root()
    runs = sorted((root / "experiments" / "neural" / args.variant).glob("seed*"))
    scored = [(json.loads((r / "result.json").read_text()), r) for r in runs if (r / "best.pt").exists()]
    if not scored:
        raise SystemExit(f"no finished runs found for {args.variant!r}")
    result, run = max(scored, key=lambda x: x[0]["validation"]["full.recall@10"])
    target = root / "artifacts" / "checkpoints" / "best.pt"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists; pass --force to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "best.pt", target)
    cfg = result["model_config"]
    test_path = run / "test_result.json"
    meta = {
        "variant": args.variant,
        "branches": [k for k, on in cfg["branches"].items() if on],
        "seed": result["seed"],
        "parameters": result["parameters"],
        "selected_by": "best validation Recall@10 across the variant's seeds",
        "best_epoch": result["best_epoch"],
        "validation": {k: v for k, v in result["validation"].items() if k.startswith("full.")},
        "test": (
            {k: v for k, v in json.loads(test_path.read_text())["test"].items() if k.startswith("full.")}
            if test_path.exists()
            else None
        ),
    }
    out = root / "artifacts" / "model_metadata" / "model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    val = meta["validation"]["full.recall@10"]
    print(f"exported {args.variant} seed {meta['seed']} (validation Recall@10 {val:.4f})")
