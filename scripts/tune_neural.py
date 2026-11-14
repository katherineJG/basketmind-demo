"""Tune BasketMind's training settings on validation (Optuna). Resumable: rerun to continue up to --trials."""

import argparse

from basketmind.training.tune import run_neural_tuning
from basketmind.utils.device import release_gpu

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=24)
    study = run_neural_tuning(ap.parse_args().trials)
    print("best", round(study.best_value, 4), study.best_params, flush=True)
    release_gpu()
