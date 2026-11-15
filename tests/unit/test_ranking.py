import math

import numpy as np
import pytest
import torch

from basketmind.evaluation.ranking import RankingMetrics

SCORES = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])  # ranking order: items 0,1,2,3,4
TARGETS = torch.tensor([[False, True, False, False, True]])  # relevant: items 1 (rank 2) and 4 (rank 5)


def test_metrics_hand_computed():
    m = RankingMetrics([3, 5])
    m.update(SCORES, TARGETS)
    r = m.compute()
    assert r["recall@3"] == pytest.approx(1 / 2)
    assert r["recall@5"] == pytest.approx(1.0)
    assert r["precision@3"] == pytest.approx(1 / 3)
    dcg, idcg = 1 / math.log2(3), 1 + 1 / math.log2(3)
    assert r["ndcg@3"] == pytest.approx(dcg / idcg)
    assert r["mrr"] == pytest.approx(1 / 2)
    assert r["n_evaluated"] == 1


def test_perfect_ranking_scores_one():
    m = RankingMetrics([2])
    m.update(torch.tensor([[0.1, 0.9, 0.8]]), torch.tensor([[False, True, True]]))
    r = m.compute()
    assert r["recall@2"] == r["ndcg@2"] == r["mrr"] == pytest.approx(1.0)


def test_exclude_removes_items_from_candidates_and_targets():
    m = RankingMetrics([3])
    exclude = torch.tensor([[False, True, False, False, False]])  # item 1 (a target) is history
    m.update(SCORES, TARGETS, exclude)
    r = m.compute()
    assert r["mrr"] == pytest.approx(1 / 4)  # only item 4 left, ranked 4th among 0,2,3,4
    assert r["recall@3"] == 0.0


def _three_examples():
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.1, 0.4, 0.2, 0.3]])
    targets = torch.tensor(
        [
            [False, True, False, False, True],
            [True, False, False, False, False],
            [False, False, False, False, True],
        ]
    )
    return scores, targets


def test_per_example_values_average_back_to_the_aggregate_metrics():
    scores, targets = _three_examples()
    m = RankingMetrics([3, 5], keep_examples=True)
    m.update(scores, targets, ids=np.array([10, 11, 12]))
    per, agg = m.per_example(), m.compute()
    assert per["example_id"].tolist() == [10, 11, 12]
    for name in ("recall@3", "recall@5", "precision@3", "ndcg@3", "ndcg@5", "mrr"):
        assert per[name].mean() == pytest.approx(agg[name]), name
    assert per["recall@3"][0] == pytest.approx(0.5)  # the hand-computed example from above
    assert per["top"].shape == (3, 5) and per["top"][0].tolist() == [0, 1, 2, 3, 4]


def test_ids_stay_aligned_when_examples_are_skipped():
    scores, targets = _three_examples()
    exclude = torch.zeros_like(targets)
    exclude[1, 0] = True  # example 11's only target is history: it is skipped under exclude_history
    m = RankingMetrics([3], keep_examples=True)
    m.update(scores, targets, exclude, ids=np.array([10, 11, 12]))
    per = m.per_example()
    assert per["example_id"].tolist() == [10, 12] and m.compute()["n_skipped"] == 1
    assert len(per["top"]) == len(per["mrr"]) == 2
    assert per["top"][1].tolist()[:1] == [0]  # example 12's best-scored candidate is item 0


def test_per_example_mode_needs_ids_and_is_off_by_default():
    scores, targets = _three_examples()
    with pytest.raises(ValueError, match="one id per row"):
        RankingMetrics([3], keep_examples=True).update(scores, targets)
    plain = RankingMetrics([3])
    plain.update(scores, targets)  # the default path needs no ids and keeps nothing
    with pytest.raises(ValueError, match="keep_examples"):
        plain.per_example()
    assert plain._rows["mrr"] == []


def test_examples_without_targets_are_skipped_and_counted():
    m = RankingMetrics([3])
    m.update(SCORES, TARGETS, exclude=TARGETS.clone())  # every target excluded
    assert m.compute()["n_skipped"] == 1 and m.n == 0
