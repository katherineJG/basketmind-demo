"""Write docs/quantity_head.md from experiments/analysis/quantity_{validation,test}.parquet.

Run scripts/evaluate_quantity.py first (validation is required; test is included if it exists).
"""

import pandas as pd

from basketmind.evaluation.quantity import render
from basketmind.utils.config import project_root

if __name__ == "__main__":
    root = project_root()
    ana = root / "experiments" / "analysis"
    frames = {
        s: pd.read_parquet(ana / f"quantity_{s}.parquet")
        for s in ("validation", "test")
        if (ana / f"quantity_{s}.parquet").exists()
    }
    if "validation" not in frames:
        raise SystemExit("run scripts/evaluate_quantity.py --split validation first")
    (root / "docs" / "quantity_head.md").write_text(render(frames), encoding="utf-8")
    print("wrote docs/quantity_head.md for", list(frames))
