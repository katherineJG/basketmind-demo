"""Tune the GRU baseline on validation (Optuna). Resumable: rerun to continue up to --trials."""

import argparse

from basketmind.baselines.tune import run_tuning

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    s = run_tuning(ap.parse_args().trials)
    print("best", round(s.best_value, 4), s.best_params)
