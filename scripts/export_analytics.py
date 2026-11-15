"""Write artifacts/analytics/{cancellations,performance}.json for the API and dashboard.

Needs data/processed/ (run_m1.py), experiments/analysis/per_example_test.parquet (collect_per_example.py)
and the experiment results. Nothing is trained or evaluated here: it only summarises existing results.
"""

import json

import pandas as pd

from basketmind.baselines.task import load_task
from basketmind.evaluation.final_report import aggregate, baseline_rows, load_neural
from basketmind.evaluation.slices import assign_slices
from basketmind.inference.analytics import cancellation_payload, performance_payload
from basketmind.inference.service import load_descriptions
from basketmind.utils.config import load_config, project_root

VARIANTS = ["abl_full", "abl_graph_customer"]

if __name__ == "__main__":
    root = project_root()
    proc, out = root / "data" / "processed", root / "artifacts" / "analytics"
    out.mkdir(parents=True, exist_ok=True)

    cols = ["CustomerID", "StockCode", "Quantity", "InvoiceDate", "InvoiceNo", "has_customer"]
    purchases = pd.read_parquet(proc / "purchase_events.parquet", columns=cols)
    cancellations = pd.read_parquet(proc / "cancellation_events.parquet", columns=cols)
    regex = load_config("data")["product_code_regex"]
    canc = cancellation_payload(purchases, cancellations, regex, load_descriptions(proc))
    (out / "cancellations.json").write_text(json.dumps(canc, indent=2), encoding="utf-8")

    task = load_task(proc)
    baskets = pd.read_parquet(proc / "baskets.parquet")
    slices = assign_slices(task, baskets.set_index("InvoiceNo")["Country"].to_dict(), "test")
    analysis = root / "experiments" / "analysis"
    rows = pd.read_parquet(analysis / "per_example_test.parquet")
    extra = analysis / "per_example_repeat.parquet"  # scripts/run_repeat_baselines.py; optional
    if extra.exists():
        pf = pd.read_parquet(extra)
        rows = pd.concat([rows, pf[pf["model"] == "personal_frequency"]], ignore_index=True)
    neural = aggregate(load_neural(root / "experiments" / "neural", VARIANTS))
    frames = [
        baseline_rows(root / "experiments" / "baselines" / "summary.csv").rename(
            columns={"variant": "model"}
        ),
        neural.rename(columns={"variant": "model"}),
    ]
    val = pd.concat(frames)
    val = val[(val["split"] == "validation") & (val["policy"] == "full")]
    validation = {
        r["model"]: {"recall@10": float(r["recall@10"]), "std": float(r["recall@10_std"])}
        for _, r in val.iterrows()
    }
    if extra.exists():  # validation score the setting was chosen on (deterministic, so no seed spread)
        chosen = json.loads((analysis / "repeat_baselines.json").read_text())["personal_frequency"]
        validation["personal_frequency"] = {"recall@10": chosen["selected"]["best_val"], "std": 0.0}
    perf = performance_payload(rows, slices, validation, task.popularity, task.n_items)
    (out / "performance.json").write_text(json.dumps(perf, indent=2), encoding="utf-8")
    print(f"wrote {out / 'cancellations.json'} and {out / 'performance.json'}")
