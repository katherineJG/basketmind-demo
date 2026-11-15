"""Write docs/evaluation_rigour.md from experiments/analysis/ (run collect_per_example.py first)."""

import json

import pandas as pd

from basketmind.baselines.task import load_task
from basketmind.evaluation.rigour import render
from basketmind.evaluation.slices import assign_slices
from basketmind.utils.config import project_root

if __name__ == "__main__":
    root = project_root()
    out = root / "experiments" / "analysis"
    rows = pd.read_parquet(out / "per_example_test.parquet")
    consistency = json.loads((out / "consistency.json").read_text())
    task = load_task(root / "data" / "processed")
    baskets = pd.read_parquet(root / "data" / "processed" / "baskets.parquet")
    slices = assign_slices(task, baskets.set_index("InvoiceNo")["Country"].to_dict(), "test")
    report = render(rows, slices, consistency, task.popularity, task.n_items)
    (root / "docs" / "evaluation_rigour.md").write_text(report, encoding="utf-8")
    print("wrote docs/evaluation_rigour.md")
