"""Write artifacts/demo/ (the public demo's bundle): python scripts/export_demo.py [--device cpu]

Needs data/processed, the exported checkpoint (scripts/export_model.py) and artifacts/analytics
(scripts/export_analytics.py). It runs the real model once per customer, so it takes a few minutes.
"""

import argparse

from basketmind.inference.demo_export import write_bundle
from basketmind.inference.service import BasketMindService
from basketmind.utils.config import project_root
from basketmind.utils.logging import get_logger

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()
    log = get_logger("export_demo")
    root = project_root()
    service = BasketMindService.from_artifacts(root, device=args.device)
    sizes = write_bundle(service, root / "artifacts" / "demo", args.k, log.info)
    for name, size in sizes.items():
        print(f"{name:20s} {size / 1e6:7.2f} MB")
