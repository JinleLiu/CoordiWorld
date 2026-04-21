"""Robustness perturbations and ranking stability diagnostics."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from coordiworld.attribution.masking import mask_entity_tokens
from coordiworld.evaluation.ranking_metrics import kendall_correlation, select_top1_index


@dataclass(frozen=True)
class EvidenceDropoutResult:
    agent_tensor: list[list[float]]
    agent_mask: list[int]
    dropped_indices: list[int]


@dataclass(frozen=True)
class RankingStability:
    top1_unchanged: bool
    kendall: float
    mean_abs_score_delta: float
    baseline_top1: int
    perturbed_top1: int


def apply_confidence_noise(
    agent_tensor: Sequence[Sequence[float]],
    agent_mask: Sequence[int] | None = None,
    *,
    noise_std: float = 0.05,
    seed: int = 0,
    confidence_index: int = 7,
) -> list[list[float]]:
    """Apply deterministic Gaussian confidence noise to active entity tokens."""
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    rng = random.Random(seed)
    mask = _mask(agent_mask, len(agent_tensor))
    perturbed = _copy_2d(agent_tensor)
    for index, token in enumerate(perturbed):
        if not mask[index] or confidence_index >= len(token):
            continue
        token[confidence_index] = _clip01(token[confidence_index] + rng.gauss(0.0, noise_std))
    return perturbed


def mask_provenance_channels(
    agent_tensor: Sequence[Sequence[float]],
    agent_mask: Sequence[int] | None = None,
    *,
    source_count_index: int = 13,
    ambiguity_count_index: int = 14,
) -> list[list[float]]:
    """Mask provenance/ambiguity token channels without changing geometry."""
    mask = _mask(agent_mask, len(agent_tensor))
    masked = _copy_2d(agent_tensor)
    for index, token in enumerate(masked):
        if not mask[index]:
            continue
        if source_count_index < len(token):
            token[source_count_index] = 0.0
        if ambiguity_count_index < len(token):
            token[ambiguity_count_index] = 0.0
    return masked


def apply_evidence_dropout(
    agent_tensor: Sequence[Sequence[float]],
    agent_mask: Sequence[int] | None = None,
    *,
    dropout_prob: float = 0.1,
    seed: int = 0,
) -> EvidenceDropoutResult:
    """Drop active entity evidence deterministically."""
    if dropout_prob < 0.0 or dropout_prob > 1.0:
        raise ValueError("dropout_prob must be in [0, 1]")
    rng = random.Random(seed)
    mask = _mask(agent_mask, len(agent_tensor))
    dropped: list[int] = []
    for index, active in enumerate(mask):
        if active and rng.random() < dropout_prob:
            dropped.append(index)
    result = mask_entity_tokens(agent_tensor, mask, dropped)
    return EvidenceDropoutResult(
        agent_tensor=result.agent_tensor,
        agent_mask=result.agent_mask,
        dropped_indices=dropped,
    )


def compute_ranking_stability(
    baseline_scores: Sequence[float],
    perturbed_scores: Sequence[float],
    *,
    lower_is_better: bool = True,
) -> RankingStability:
    """Compare candidate ranking before and after perturbation."""
    if len(baseline_scores) != len(perturbed_scores):
        raise ValueError("baseline_scores and perturbed_scores must have equal length")
    if not baseline_scores:
        raise ValueError("scores must not be empty")
    baseline_top1 = select_top1_index(
        baseline_scores,
        prediction_lower_is_better=lower_is_better,
    )
    perturbed_top1 = select_top1_index(
        perturbed_scores,
        prediction_lower_is_better=lower_is_better,
    )
    mean_abs_delta = sum(
        abs(float(after) - float(before))
        for before, after in zip(baseline_scores, perturbed_scores)
    ) / len(baseline_scores)
    return RankingStability(
        top1_unchanged=baseline_top1 == perturbed_top1,
        kendall=kendall_correlation(
            perturbed_scores,
            baseline_scores,
            prediction_lower_is_better=lower_is_better,
            target_lower_is_better=lower_is_better,
        ),
        mean_abs_score_delta=mean_abs_delta,
        baseline_top1=baseline_top1,
        perturbed_top1=perturbed_top1,
    )


def _copy_2d(values: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(item) for item in row] for row in values]


def _mask(mask: Sequence[int] | None, length: int) -> list[int]:
    if mask is None:
        return [1 for _ in range(length)]
    if len(mask) != length:
        raise ValueError("mask length must match agent_tensor")
    return [1 if bool(value) else 0 for value in mask]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
