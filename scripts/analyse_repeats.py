"""Write docs/repeat_purchases.md: where each model's test hits come from (repeat purchases versus new items).

Needs experiments/analysis/per_example_test.parquet (scripts/collect_per_example.py). If
experiments/analysis/per_example_repeat.parquet exists (scripts/run_repeat_baselines.py), it is included too.
Analysis only: it reads finished results and changes no model.
"""

import pandas as pd

from basketmind.baselines.task import load_task
from basketmind.evaluation.repeat import render, repeat_columns, seed_mean, target_kinds
from basketmind.utils.config import project_root

if __name__ == "__main__":
    root = project_root()
    out = root / "experiments" / "analysis"
    files = [out / "per_example_test.parquet", out / "per_example_repeat.parquet"]
    rows = pd.concat([pd.read_parquet(f) for f in files if f.exists()], ignore_index=True)
    task = load_task(root / "data" / "processed")
    kinds = target_kinds(task, "test")
    customers = task.examples["test"]["CustomerID"].rename_axis("example_id")
    avg = seed_mean(repeat_columns(rows, kinds))
    (root / "docs" / "repeat_purchases.md").write_text(render(avg, kinds, customers), encoding="utf-8")
    print("wrote docs/repeat_purchases.md for", sorted(avg["model"].unique()))
