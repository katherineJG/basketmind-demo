"""Train and evaluate all baseline recommenders; writes docs/baseline_report.md."""

from basketmind.baselines.run import run_baselines

if __name__ == "__main__":
    run_baselines()
