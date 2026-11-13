"""Tune a baseline on validation (Optuna). Resumable: rerun to continue up to --trials."""

import argparse

from basketmind.baselines.tune import SPACES, run_tuning

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(SPACES), required=True)
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()
    s = run_tuning(args.model, args.trials)
    print("best", round(s.best_value, 4), s.best_params)
