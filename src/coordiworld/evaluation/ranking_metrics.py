"""Ranking metrics for shared candidate-set evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RankingMetrics:
    spearman: float
    kendall: float
    ndcg_at_3: float
    top1_collision: float
    top1_violation: float
    selected_index: int


def spearman_correlation(
    predicted_scores: Sequence[float],
    target_scores: Sequence[float],
    *,
    prediction_lower_is_better: bool = True,
    target_lower_is_better: bool = True,
) -> float:
    """Compute Spearman rank correlation with average ranks for ties."""
    predicted, target = _aligned_scores(
        predicted_scores,
        target_scores,
        prediction_lower_is_better=prediction_lower_is_better,
        target_lower_is_better=target_lower_is_better,
    )
    return _pearson(_average_ranks(predicted), _average_ranks(target))


def kendall_correlation(
    predicted_scores: Sequence[float],
    target_scores: Sequence[float],
    *,
    prediction_lower_is_better: bool = True,
    target_lower_is_better: bool = True,
) -> float:
    """Compute Kendall tau-b for candidate ordering."""
    predicted, target = _aligned_scores(
        predicted_scores,
        target_scores,
        prediction_lower_is_better=prediction_lower_is_better,
        target_lower_is_better=target_lower_is_better,
    )
    concordant = 0
    discordant = 0
    predicted_ties = 0
    target_ties = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            predicted_cmp = _sign(predicted[left] - predicted[right])
            target_cmp = _sign(target[left] - target[right])
            if predicted_cmp == 0 and target_cmp == 0:
                continue
            if predicted_cmp == 0:
                predicted_ties += 1
                continue
            if target_cmp == 0:
                target_ties += 1
                continue
            if predicted_cmp == target_cmp:
                concordant += 1
            else:
                discordant += 1

    denominator = math.sqrt(
        (concordant + discordant + predicted_ties)
        * (concordant + discordant + target_ties)
    )
    if denominator == 0:
        return 0.0
    return (concordant - discordant) / denominator


def ndcg_at_k(
    predicted_scores: Sequence[float],
    relevance_scores: Sequence[float],
    *,
    k: int = 3,
    prediction_lower_is_better: bool = True,
    relevance_lower_is_better: bool = False,
) -> float:
    """Compute NDCG@k for a shared candidate set."""
    _validate_equal_length(predicted_scores, relevance_scores)
    if k <= 0:
        raise ValueError("k must be > 0")
    relevance = _relevance_values(relevance_scores, lower_is_better=relevance_lower_is_better)
    order = _order_indices(predicted_scores, lower_is_better=prediction_lower_is_better)
    ideal_order = sorted(range(len(relevance)), key=lambda index: (-relevance[index], index))
    cutoff = min(k, len(relevance))
    dcg = _dcg([relevance[index] for index in order[:cutoff]])
    ideal_dcg = _dcg([relevance[index] for index in ideal_order[:cutoff]])
    return 0.0 if ideal_dcg == 0 else dcg / ideal_dcg


def top1_collision(
    predicted_scores: Sequence[float],
    collision_labels: Sequence[bool | int | float],
    *,
    prediction_lower_is_better: bool = True,
) -> float:
    """Return whether the selected top-1 candidate collides."""
    selected = select_top1_index(
        predicted_scores,
        prediction_lower_is_better=prediction_lower_is_better,
    )
    return float(bool(collision_labels[selected]))


def top1_violation(
    predicted_scores: Sequence[float],
    violation_labels: Sequence[bool | int | float],
    *,
    prediction_lower_is_better: bool = True,
) -> float:
    """Return whether the selected top-1 candidate violates map rules."""
    selected = select_top1_index(
        predicted_scores,
        prediction_lower_is_better=prediction_lower_is_better,
    )
    return float(bool(violation_labels[selected]))


def compute_ranking_metrics(
    predicted_scores: Sequence[float],
    target_scores: Sequence[float],
    collision_labels: Sequence[bool | int | float],
    violation_labels: Sequence[bool | int | float],
    *,
    prediction_lower_is_better: bool = True,
    target_lower_is_better: bool = True,
    ndcg_k: int = 3,
) -> RankingMetrics:
    """Compute evaluator ranking and selected top-1 safety metrics."""
    _validate_equal_length(predicted_scores, target_scores)
    _validate_equal_length(predicted_scores, collision_labels)
    _validate_equal_length(predicted_scores, violation_labels)
    selected = select_top1_index(
        predicted_scores,
        prediction_lower_is_better=prediction_lower_is_better,
    )
    return RankingMetrics(
        spearman=spearman_correlation(
            predicted_scores,
            target_scores,
            prediction_lower_is_better=prediction_lower_is_better,
            target_lower_is_better=target_lower_is_better,
        ),
        kendall=kendall_correlation(
            predicted_scores,
            target_scores,
            prediction_lower_is_better=prediction_lower_is_better,
            target_lower_is_better=target_lower_is_better,
        ),
        ndcg_at_3=ndcg_at_k(
            predicted_scores,
            target_scores,
            k=ndcg_k,
            prediction_lower_is_better=prediction_lower_is_better,
            relevance_lower_is_better=target_lower_is_better,
        ),
        top1_collision=float(bool(collision_labels[selected])),
        top1_violation=float(bool(violation_labels[selected])),
        selected_index=selected,
    )


def select_top1_index(
    predicted_scores: Sequence[float],
    *,
    prediction_lower_is_better: bool = True,
) -> int:
    if not predicted_scores:
        raise ValueError("predicted_scores must not be empty")
    _require_finite(predicted_scores, "predicted_scores")
    key = (
        (lambda index: (float(predicted_scores[index]), index))
        if prediction_lower_is_better
        else (lambda index: (-float(predicted_scores[index]), index))
    )
    return min(range(len(predicted_scores)), key=key)


def _aligned_scores(
    predicted_scores: Sequence[float],
    target_scores: Sequence[float],
    *,
    prediction_lower_is_better: bool,
    target_lower_is_better: bool,
) -> tuple[list[float], list[float]]:
    _validate_equal_length(predicted_scores, target_scores)
    _require_finite(predicted_scores, "predicted_scores")
    _require_finite(target_scores, "target_scores")
    predicted = [float(score) for score in predicted_scores]
    target = [float(score) for score in target_scores]
    if prediction_lower_is_better:
        predicted = [-score for score in predicted]
    if target_lower_is_better:
        target = [-score for score in target]
    return predicted, target


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0 for _ in values]
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for sorted_index in range(cursor, end):
            ranks[indexed[sorted_index][0]] = average_rank
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    _validate_equal_length(left, right)
    if len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_var * right_var)
    return 0.0 if denominator == 0 else numerator / denominator


def _relevance_values(scores: Sequence[float], *, lower_is_better: bool) -> list[float]:
    _require_finite(scores, "relevance_scores")
    values = [float(score) for score in scores]
    if lower_is_better:
        maximum = max(values)
        values = [maximum - value for value in values]
    minimum = min(values)
    if minimum < 0:
        values = [value - minimum for value in values]
    return values


def _order_indices(scores: Sequence[float], *, lower_is_better: bool) -> list[int]:
    _require_finite(scores, "predicted_scores")
    return sorted(
        range(len(scores)),
        key=(
            (lambda index: (float(scores[index]), index))
            if lower_is_better
            else (lambda index: (-float(scores[index]), index))
        ),
    )


def _dcg(relevance: Sequence[float]) -> float:
    return sum(
        (2.0**value - 1.0) / math.log2(index + 2.0)
        for index, value in enumerate(relevance)
    )


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _validate_equal_length(left: Sequence[object], right: Sequence[object]) -> None:
    if len(left) != len(right):
        raise ValueError("metric inputs must have equal length")
    if not left:
        raise ValueError("metric inputs must not be empty")


def _require_finite(values: Sequence[float], name: str) -> None:
    for value in values:
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must contain finite values")
