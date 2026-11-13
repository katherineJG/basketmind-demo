"""Run the M1 data pipeline. Use --refresh to re-read the xlsx instead of the parquet cache."""

import argparse

from basketmind.data.pipeline import run_m1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    run_m1(refresh=ap.parse_args().refresh)
