"""Synthetic tests for post-hoc entity attribution."""

from __future__ import annotations

import pytest

from coordiworld.attribution.counterfactual import compute_entity_attributions
from coordiworld.attribution.masking import mask_entity_tokens, select_nearby_entities
from coordiworld.evaluation.auditability import entity_recall_at_k, risk_drop_at_k


def make_agent_tensor() -> list[list[float]]:
    return [
        [2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.9, 1.0, 0.1, 0.0, 0.1, 1.0, 2.0, 0.0, 0.9],
        [5.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.8, 1.0, 0.1, 0.0, 0.1, 1.0, 1.0, 0.0, 0.2],
        [20.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.7, 1.0, 0.1, 0.0, 0.1, 1.0, 1.0, 0.0, 10.0],
    ]


def mock_scorer(_trajectory, agent_tensor, agent_mask) -> float:
    return sum(token[15] for token, active in zip(agent_tensor, agent_mask) if active)


def test_mask_entity_tokens_masks_copy_not_original() -> None:
    original = make_agent_tensor()
    result = mask_entity_tokens(original, [1, 1, 1], [1])

    assert result.masked_indices == [1]
    assert result.agent_mask == [1, 0, 1]
    assert result.agent_tensor[1] == [0.0] * len(original[1])
    assert original[1][15] == 0.2


def test_nearby_entity_selection_uses_distance_and_risk_heuristic() -> None:
    selected = select_nearby_entities(make_agent_tensor(), [1, 1, 1], max_entities=2)

    assert selected == [0, 2]


def test_counterfactual_entity_attribution_recomputes_selected_j() -> None:
    attributions = compute_entity_attributions(
        selected_trajectory=[[0.0, 0.0, 0.0]],
        agent_tensor=make_agent_tensor(),
        agent_mask=[1, 1, 1],
        scorer=mock_scorer,
    )

    assert [item.entity_index for item in attributions] == [2, 0, 1]
    assert attributions[0].baseline_score == 11.1
    assert attributions[0].masked_score == 1.1
    assert attributions[0].delta == 10.0


def test_auditability_metrics_riskdrop_and_entity_recall() -> None:
    attributions = compute_entity_attributions(
        selected_trajectory=None,
        agent_tensor=make_agent_tensor(),
        agent_mask=[1, 1, 1],
        scorer=mock_scorer,
    )

    assert risk_drop_at_k(attributions, k=2) == pytest.approx(10.9)
    assert entity_recall_at_k(attributions, [2, 1], k=2) == 0.5
