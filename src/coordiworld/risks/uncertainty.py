"""Predictive uncertainty risk component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from coordiworld.risks.geometry import clip01


@dataclass(frozen=True)
class PredictiveUncertaintyResult:
    existence_weighted_trace: float
    u_bar: float


def compute_predictive_uncertainty(
    covariance: Sequence[Sequence[Sequence[float] | Sequence[Sequence[float]]]],
    *,
    existence_probabilities: Sequence[Sequence[float]] | None = None,
    u95: float = 4.0,
) -> PredictiveUncertaintyResult:
    """Compute existence-weighted trace covariance and normalize by u95."""
    if u95 <= 0:
        raise ValueError("u95 must be > 0")
    weighted_sum = 0.0
    weight_sum = 0.0
    for agent_index, agent_covariance in enumerate(covariance):
        for step_index, step_covariance in enumerate(agent_covariance):
            trace = trace_covariance(step_covariance)
            existence = _existence(existence_probabilities, agent_index, step_index)
            weighted_sum += existence * trace
            weight_sum += existence
    raw = 0.0 if weight_sum == 0 else weighted_sum / weight_sum
    return PredictiveUncertaintyResult(
        existence_weighted_trace=raw,
        u_bar=clip01(raw / u95),
    )


def trace_covariance(covariance: Sequence[float] | Sequence[Sequence[float]]) -> float:
    """Return planar trace from [var_x, var_y] or [[xx,xy],[yx,yy]]."""
    if len(covariance) == 2 and all(isinstance(value, (int, float)) for value in covariance):
        return float(covariance[0]) + float(covariance[1])
    matrix = covariance
    if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
        raise ValueError("covariance must be [var_x,var_y] or 2x2")
    return float(matrix[0][0]) + float(matrix[1][1])


def _existence(
    existence_probabilities: Sequence[Sequence[float]] | None,
    agent_index: int,
    step_index: int,
) -> float:
    if existence_probabilities is None:
        return 1.0
    return clip01(float(existence_probabilities[agent_index][step_index]))
