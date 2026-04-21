"""Audit-only counterfactual entity attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from coordiworld.attribution.masking import mask_entity_tokens

CounterfactualScorer = Callable[[Any, Sequence[Sequence[float]], Sequence[int]], float]


@dataclass(frozen=True)
class EntityAttribution:
    entity_index: int
    baseline_score: float
    masked_score: float
    delta: float
    abs_delta: float


def recompute_selected_trajectory_j(
    selected_trajectory: Any,
    agent_tensor: Sequence[Sequence[float]],
    agent_mask: Sequence[int],
    scorer: CounterfactualScorer,
) -> float:
    """Recompute J for the selected trajectory with a supplied scorer."""
    return float(scorer(selected_trajectory, agent_tensor, agent_mask))


def compute_entity_attributions(
    selected_trajectory: Any,
    agent_tensor: Sequence[Sequence[float]],
    agent_mask: Sequence[int],
    scorer: CounterfactualScorer,
    *,
    entity_indices: Sequence[int] | None = None,
) -> list[EntityAttribution]:
    """Compute audit-only entity deltas.

    delta_i = J(tau) - J(tau | mask(entity_i)).
    A positive delta means masking the entity reduced the selected trajectory risk.
    """
    baseline = recompute_selected_trajectory_j(
        selected_trajectory,
        agent_tensor,
        agent_mask,
        scorer,
    )
    indices = (
        [index for index, active in enumerate(agent_mask) if active]
        if entity_indices is None
        else list(entity_indices)
    )
    attributions: list[EntityAttribution] = []
    for entity_index in indices:
        masked = mask_entity_tokens(agent_tensor, agent_mask, [entity_index])
        masked_score = recompute_selected_trajectory_j(
            selected_trajectory,
            masked.agent_tensor,
            masked.agent_mask,
            scorer,
        )
        delta = baseline - masked_score
        attributions.append(
            EntityAttribution(
                entity_index=int(entity_index),
                baseline_score=baseline,
                masked_score=masked_score,
                delta=delta,
                abs_delta=abs(delta),
            )
        )
    return rank_entity_attributions(attributions)


def rank_entity_attributions(attributions: Sequence[EntityAttribution]) -> list[EntityAttribution]:
    """Rank attributions by absolute delta, then stable entity index."""
    return sorted(attributions, key=lambda item: (-item.abs_delta, item.entity_index))
