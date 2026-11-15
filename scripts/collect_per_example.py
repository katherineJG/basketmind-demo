"""Collect per-example test results for every model (analysis only; see evaluation/per_example.py).

Writes experiments/analysis/per_example_test.parquet and experiments/analysis/consistency.json.
"""

import json

from basketmind.evaluation.per_example import collect_all
from basketmind.utils.config import project_root
from basketmind.utils.device import release_gpu

if __name__ == "__main__":
    root = project_root()
    out = root / "experiments" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rows, consistency = collect_all(root, "test")
    rows.to_parquet(out / "per_example_test.parquet")
    (out / "consistency.json").write_text(json.dumps(consistency, indent=2), encoding="utf-8")
    worst = max(abs(r["diff"]) for r in consistency["rows"])
    print(
        f"{len(rows):,} rows; largest mean Recall@10 change versus the reported runs: {worst:.4f}", flush=True
    )
    release_gpu()
