"""Entity masking utilities for audit-only counterfactual attribution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Sequence


@dataclass(frozen=True)
class EntityMaskResult:
    agent_tensor: list[list[float]]
    agent_mask: list[int]
    masked_indices: list[int]


def mask_entity_tokens(
    agent_tensor: Sequence[Sequence[float]],
    agent_mask: Sequence[int] | None,
    entity_indices: Sequence[int],
    *,
    fill_value: float = 0.0,
) -> EntityMaskResult:
    """Return a copy of agent tokens with selected entities masked out."""
    tokens = _copy_2d(agent_tensor)
    mask = _copy_mask(agent_mask, len(tokens))
    selected = sorted(set(int(index) for index in entity_indices))
    for index in selected:
        if index < 0 or index >= len(tokens):
            raise ValueError(f"entity index out of range: {index}")
        tokens[index] = [float(fill_value) for _ in tokens[index]]
        mask[index] = 0
    return EntityMaskResult(agent_tensor=tokens, agent_mask=mask, masked_indices=selected)


def select_nearby_entities(
    agent_tensor: Sequence[Sequence[float]],
    agent_mask: Sequence[int] | None = None,
    *,
    radius: float | None = None,
    max_entities: int | None = None,
    x_index: int = 0,
    y_index: int = 1,
    risk_index: int = 15,
) -> list[int]:
    """Select active nearby entities by distance and a risk heuristic."""
    if radius is not None and radius < 0:
        raise ValueError("radius must be non-negative")
    if max_entities is not None and max_entities <= 0:
        raise ValueError("max_entities must be > 0")
    mask = _copy_mask(agent_mask, len(agent_tensor))
    candidates: list[tuple[float, float, int]] = []
    for index, token in enumerate(agent_tensor):
        if not mask[index]:
            continue
        x = _number(token[x_index], f"agent_tensor[{index}][{x_index}]")
        y = _number(token[y_index], f"agent_tensor[{index}][{y_index}]")
        distance = math.hypot(x, y)
        if radius is not None and distance > radius:
            continue
        risk_score = _optional_number(token, risk_index)
        effective_distance = distance / (1.0 + max(0.0, risk_score))
        candidates.append((effective_distance, distance, index))
    selected = [index for _, _, index in sorted(candidates)]
    if max_entities is not None:
        return selected[:max_entities]
    return selected


def _copy_2d(values: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(item) for item in row] for row in values]


def _copy_mask(mask: Sequence[int] | None, length: int) -> list[int]:
    if mask is None:
        return [1 for _ in range(length)]
    if len(mask) != length:
        raise ValueError("agent_mask length must match agent_tensor")
    return [1 if bool(value) else 0 for value in mask]


def _optional_number(values: Sequence[float], index: int) -> float:
    if index < 0 or index >= len(values):
        return 0.0
    return _number(values[index], f"token[{index}]")


def _number(value: object, path: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{path} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{path} must be finite")
    return float(value)
