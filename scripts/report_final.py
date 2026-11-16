"""Write docs/final_comparison.md from the neural runs and the baseline summary.

python scripts/report_final.py --variants abl_full,abl_graph_customer
"""

import argparse

import pandas as pd

from basketmind.evaluation.final_report import (
    aggregate,
    baseline_rows,
    load_neural,
    personal_frequency_rows,
    render,
    select_variant,
)
from basketmind.utils.config import project_root

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="comma list of neural variant folders to compare")
    args = ap.parse_args()
    root = project_root()
    variants = args.variants.split(",")
    agg = aggregate(load_neural(root / "experiments" / "neural", variants))
    selected = select_variant(agg, variants)
    base = baseline_rows(root / "experiments" / "baselines" / "summary.csv")
    extra = (
        root / "experiments" / "analysis" / "repeat_baselines.json"
    )  # scripts/run_repeat_baselines.py; optional
    if extra.exists():
        base = pd.concat([base, personal_frequency_rows(extra, base)], ignore_index=True)
    (root / "docs" / "final_comparison.md").write_text(render(agg, base, selected), encoding="utf-8")
    print("selected on validation:", selected)
