"""Auditability metrics for post-hoc entity attribution."""

from __future__ import annotations

from typing import Sequence

from coordiworld.attribution.counterfactual import EntityAttribution


def risk_drop_at_k(
    attributions: Sequence[EntityAttribution] | Sequence[float],
    *,
    k: int,
) -> float:
    """Compute RiskDrop@K as the sum of positive top-K attribution deltas."""
    if k <= 0:
        raise ValueError("k must be > 0")
    deltas = _deltas(attributions)
    ranked = sorted(deltas, reverse=True)
    return sum(delta for delta in ranked[:k] if delta > 0.0)


def entity_recall_at_k(
    attributions: Sequence[EntityAttribution],
    ground_truth_entity_indices: Sequence[int],
    *,
    k: int,
) -> float:
    """Compute EntityRecall@K over audit-relevant ground-truth entities."""
    if k <= 0:
        raise ValueError("k must be > 0")
    ground_truth = {int(index) for index in ground_truth_entity_indices}
    if not ground_truth:
        return 0.0
    ranked = sorted(attributions, key=lambda item: (-item.abs_delta, item.entity_index))
    selected = {item.entity_index for item in ranked[:k]}
    return len(selected & ground_truth) / len(ground_truth)


def _deltas(attributions: Sequence[EntityAttribution] | Sequence[float]) -> list[float]:
    values: list[float] = []
    for item in attributions:
        if isinstance(item, EntityAttribution):
            values.append(float(item.delta))
        else:
            values.append(float(item))
    return values
