"""Evaluate the TEST split once for finished runs, using each run's best-validation checkpoint.

    python scripts/evaluate_test.py --runs abl_full:42,abl_full:43,abl_graph_customer:42

Each run is evaluated at most once: if its test_result.json exists the script stops instead of overwriting it.
"""

import argparse

from basketmind.training.final_eval import evaluate_test_once
from basketmind.utils.config import project_root

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of name:seed, e.g. abl_full:42,abl_full:43")
    runs = [(n, int(s)) for n, s in (r.split(":") for r in ap.parse_args().runs.split(","))]
    for key, res in evaluate_test_once(project_root(), runs).items():
        t = res["test"]
        print(f"{key}: test recall@10={t['full.recall@10']:.4f} mrr={t['full.mrr']:.4f}", flush=True)
