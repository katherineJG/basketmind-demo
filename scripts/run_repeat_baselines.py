"""Fit the personal-frequency baselines on VALIDATION, then evaluate them on test once.

Writes experiments/analysis/per_example_repeat.parquet (per-example test results, same layout as
per_example_test.parquet) and experiments/analysis/repeat_baselines.json (validation grids, settings, means).
Refuses to run twice: the test split is evaluated once per model. Then run scripts/analyse_repeats.py.
"""

import json

import pandas as pd

from basketmind.baselines.models import PersonalFrequency
from basketmind.baselines.run import evaluate_model
from basketmind.baselines.task import load_task
from basketmind.evaluation.per_example import POLICIES, _frame, baseline_per_example
from basketmind.utils.config import load_config, project_root
from basketmind.utils.device import get_device, release_gpu
from basketmind.utils.logging import get_logger

log = get_logger("repeat_baselines")
DECAYS = (1.0, 0.9, 0.8, 0.7, 0.5)  # 1.0 = plain purchase counts; smaller = recent baskets count for more
CF_WEIGHTS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)

if __name__ == "__main__":
    root = project_root()
    out = root / "experiments" / "analysis"
    target = out / "per_example_repeat.parquet"
    if target.exists():
        raise SystemExit(
            f"{target.name} exists: these models were already evaluated on test (once per model)"
        )
    train_cfg = load_config("training", root / "configs")
    ks = load_config("evaluation", root / "configs")["recommendation_k"]
    device = get_device(train_cfg["device"])
    task = load_task(root / "data" / "processed")
    seed = train_cfg["seeds"][0]

    def val(model):
        return evaluate_model(model, task, "validation", "full", ks, device)["recall@10"]

    models = [
        PersonalFrequency("personal_frequency", DECAYS, (0.0,)),
        PersonalFrequency("personal_frequency_cf", DECAYS, CF_WEIGHTS),
    ]
    frames, summary = [], {}
    for model in models:
        info = model.fit(task, device, seed, val)
        log.info(
            "%s: decay %.1f, cf weight %.2f, validation recall@10 %.4f",
            model.name,
            model.decay,
            model.w,
            info["best_val"],
        )
        validation = {p: evaluate_model(model, task, "validation", p, ks, device) for p in POLICIES}
        per = baseline_per_example(
            model, task, "test", ks, device
        )  # the single test evaluation of this model
        test = {
            p: {k: float(v.mean()) for k, v in d.items() if k not in {"top", "example_id"}}
            for p, d in per.items()
        }
        summary[model.name] = {"selected": info, "validation": validation, "test_means": test}
        frames.append(_frame(per, model.name, seed))
        release_gpu()
    pd.concat(frames, ignore_index=True).to_parquet(target)
    (out / "repeat_baselines.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    for name, s in summary.items():
        print(
            name,
            s["selected"]["decay"],
            s["selected"]["cf_weight"],
            "val",
            round(s["selected"]["best_val"], 4),
            "test recall@10 full",
            round(s["test_means"]["full"]["recall@10"], 4),
        )
