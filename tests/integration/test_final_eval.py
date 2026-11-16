import json

import pandas as pd
import pytest

from basketmind.evaluation.final_report import (
    aggregate,
    baseline_rows,
    load_neural,
    personal_frequency_rows,
    render,
    select_variant,
)
from basketmind.training import engine
from basketmind.training.final_eval import check_ready, evaluate_test_once, run_dirs
from basketmind.utils.config import load_config
from basketmind.utils.device import get_device

DEV = get_device()
KS = [1, 2]


def trained_run(training_data, root, name, seed):
    model_cfg, train_cfg = load_config("model"), load_config("training")
    train_cfg |= {"max_epochs": 1, "batch_size": 4, "selection_metric": "recall@1"}
    d = root / "experiments" / "neural" / name / f"seed{seed}"
    engine.run_experiment(name, training_data, model_cfg, train_cfg, seed, d, DEV, KS, False, False)
    return d


def test_test_is_evaluated_once_from_the_best_checkpoint(training_data, synthetic_root):
    d = trained_run(training_data, synthetic_root, "v", 1)
    assert not (
        d / "test_result.json"
    ).exists()  # training with evaluate_test=False left the test split alone
    res = evaluate_test_once(synthetic_root, [("v", 1)], data=training_data, ks=KS, device=DEV)
    saved = json.loads((d / "test_result.json").read_text())
    assert saved["test"]["full.n_evaluated"] == res["v:1"]["test"]["full.n_evaluated"]
    assert saved["checkpoint_epoch"] == 1 and "full.recall@1" in saved["test"]

    with pytest.raises(FileExistsError, match="already evaluated"):  # a second evaluation is refused
        evaluate_test_once(synthetic_root, [("v", 1)], data=training_data, ks=KS, device=DEV)
    assert json.loads((d / "test_result.json").read_text()) == saved  # and the file is untouched


def test_nothing_runs_if_any_run_is_unready_or_already_done(training_data, synthetic_root):
    good = trained_run(training_data, synthetic_root, "a", 1)
    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        evaluate_test_once(synthetic_root, [("a", 1), ("b", 9)], data=training_data, ks=KS, device=DEV)
    assert not (good / "test_result.json").exists()  # the good run was not evaluated either
    (good / "test_result.json").write_text("{}")
    with pytest.raises(FileExistsError):
        check_ready(run_dirs(synthetic_root, [("a", 1)]))


def write_run(root, variant, seed, val_recall, test_recall):
    d = root / variant / f"seed{seed}"
    d.mkdir(parents=True)

    def block(recall):
        out = {}
        for policy in ("full", "exclude_history"):
            out |= {
                f"{policy}.{m}": recall
                for m in ("recall@5", "recall@10", "recall@20", "precision@10", "ndcg@10", "mrr")
            }
            out[f"{policy}.n_evaluated"] = 100
        return out

    (d / "result.json").write_text(json.dumps({"validation": block(val_recall)}))
    if test_recall is not None:
        (d / "test_result.json").write_text(json.dumps({"test": block(test_recall)}))


def test_selection_uses_validation_only_even_when_test_disagrees(tmp_path):
    for seed, (a, b) in enumerate([(0.10, 0.12), (0.12, 0.11)], 1):
        write_run(tmp_path, "A", seed, val_recall=a, test_recall=0.50)  # best on TEST, worse on validation
        write_run(tmp_path, "B", seed, val_recall=b, test_recall=0.01)
    agg = aggregate(load_neural(tmp_path, ["A", "B"]))
    assert select_variant(agg, ["A", "B"]) == "B"  # validation means: A 0.11, B 0.115
    row = agg[(agg["variant"] == "A") & (agg["split"] == "test") & (agg["policy"] == "full")].iloc[0]
    assert row["recall@10"] == pytest.approx(0.50) and row["seeds"] == 2


def test_runs_without_test_results_just_have_no_test_rows(tmp_path):
    write_run(tmp_path, "A", 1, 0.1, None)
    runs = load_neural(tmp_path, ["A"])
    assert set(runs["split"]) == {"validation"}


def test_report_lists_baselines_and_marks_the_selected_variant(tmp_path):
    write_run(tmp_path, "A", 1, 0.1, 0.09)
    write_run(tmp_path, "B", 1, 0.2, 0.19)
    agg = aggregate(load_neural(tmp_path, ["A", "B"]))
    cols = ["recall@5", "recall@10", "recall@20", "precision@10", "ndcg@10", "mrr"]
    rows = [
        {
            "model": "popularity",
            "split": s,
            "policy": p,
            "n_evaluated": 10,
            **{c: 0.05 for c in cols},
            **{c + "_std": 0.0 for c in cols},
        }
        for s in ("validation", "test")
        for p in ("full", "exclude_history")
    ]
    csv = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    md = render(agg, baseline_rows(csv), select_variant(agg, ["A", "B"]))
    assert "B **(selected)**" in md and "A **(selected)**" not in md and "| popularity |" in md
    assert "## Test, policy `full`" in md and "chosen as the variant with the best mean" in md


def test_personal_frequency_rows_join_the_comparison_with_the_example_counts_of_item_item_cf(tmp_path):
    write_run(tmp_path, "A", 1, 0.1, 0.09)
    agg = aggregate(load_neural(tmp_path, ["A"]))
    cols = ["recall@5", "recall@10", "recall@20", "precision@10", "ndcg@10", "mrr"]
    n = {
        ("validation", "full"): 7,
        ("validation", "exclude_history"): 5,
        ("test", "full"): 9,
        ("test", "exclude_history"): 6,
    }
    rows = [
        {"model": "item_item_cf", "split": s, "policy": p, "n_evaluated": n[(s, p)]}
        | dict.fromkeys(cols, 0.1)
        | {c + "_std": 0.0 for c in cols}
        for s, p in n
    ]
    csv = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    metrics = dict.fromkeys(cols, 0.3)
    saved = {
        "personal_frequency": {
            "validation": {p: metrics | {"n_evaluated": 99} for p in ("full", "exclude_history")},
            "test_means": {p: metrics for p in ("full", "exclude_history")},
        }
    }
    path = tmp_path / "repeat_baselines.json"
    path.write_text(json.dumps(saved))
    base = baseline_rows(csv)
    pf = personal_frequency_rows(path, base)
    assert len(pf) == 4 and set(pf["seeds"]) == {1}
    assert pf.set_index(["split", "policy"]).loc[("test", "full"), "n"] == 9  # copied from item-item CF
    assert (
        pf.set_index(["split", "policy"]).loc[("validation", "full"), "n"] == 99
    )  # its own validation count
    md = render(agg, pd.concat([base, pf], ignore_index=True), "A")
    assert "personal_frequency *(added after the course report)*" in md
    assert "this table is the corrected comparison" in md
    assert "corrected comparison" not in render(agg, base, "A")  # no note when the baseline is absent
