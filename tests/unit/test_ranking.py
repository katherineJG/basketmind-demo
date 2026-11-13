import math

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


def test_examples_without_targets_are_skipped_and_counted():
    m = RankingMetrics([3])
    m.update(SCORES, TARGETS, exclude=TARGETS.clone())  # every target excluded
    assert m.compute()["n_skipped"] == 1 and m.n == 0
