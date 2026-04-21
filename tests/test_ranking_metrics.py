"""Synthetic tests for evaluator ranking metrics."""

from __future__ import annotations

from coordiworld.evaluation.ranking_metrics import (
    compute_ranking_metrics,
    kendall_correlation,
    ndcg_at_k,
    select_top1_index,
    spearman_correlation,
    top1_collision,
    top1_violation,
)


def test_rank_correlations_are_perfect_for_matching_lower_is_better_scores() -> None:
    predicted = [0.1, 0.2, 0.4, 0.9]
    target = [0.05, 0.25, 0.5, 1.0]

    assert spearman_correlation(predicted, target) == 1.0
    assert kendall_correlation(predicted, target) == 1.0


def test_rank_correlations_are_negative_for_reversed_order() -> None:
    predicted = [0.9, 0.4, 0.2, 0.1]
    target = [0.05, 0.25, 0.5, 1.0]

    assert spearman_correlation(predicted, target) == -1.0
    assert kendall_correlation(predicted, target) == -1.0


def test_ndcg_at_3_rewards_good_top_ordering() -> None:
    predicted = [0.1, 0.2, 0.8, 0.9]
    relevance = [3.0, 2.0, 0.5, 0.0]

    assert ndcg_at_k(
        predicted,
        relevance,
        k=3,
        prediction_lower_is_better=True,
        relevance_lower_is_better=False,
    ) == 1.0


def test_top1_collision_and_violation_use_selected_candidate() -> None:
    predicted = [0.3, 0.1, 0.5]
    collisions = [1, 0, 1]
    violations = [0, 1, 0]

    assert select_top1_index(predicted) == 1
    assert top1_collision(predicted, collisions) == 0.0
    assert top1_violation(predicted, violations) == 1.0


def test_compute_ranking_metrics_bundle() -> None:
    metrics = compute_ranking_metrics(
        predicted_scores=[0.1, 0.3, 0.8, 1.0],
        target_scores=[0.0, 0.2, 0.7, 0.9],
        collision_labels=[0, 0, 1, 1],
        violation_labels=[0, 1, 0, 1],
    )

    assert metrics.selected_index == 0
    assert metrics.spearman == 1.0
    assert metrics.kendall == 1.0
    assert metrics.ndcg_at_3 > 0.99
    assert metrics.top1_collision == 0.0
    assert metrics.top1_violation == 0.0
