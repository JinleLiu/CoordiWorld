"""Deterministic association helpers for ICA-style SceneSummary generation."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence


@dataclass(frozen=True)
class AssociationMatch:
    left_index: int
    right_index: int
    cost: float


@dataclass(frozen=True)
class AssociationResult:
    matches: list[AssociationMatch]
    unmatched_left: list[int]
    unmatched_right: list[int]


CLASS_GROUPS: dict[str, frozenset[str]] = {
    "vehicle": frozenset({"vehicle", "car", "truck", "bus", "van"}),
    "pedestrian": frozenset({"pedestrian", "person"}),
    "cyclist": frozenset({"cyclist", "bicycle", "bike"}),
    "unknown": frozenset({"", "unknown"}),
}


def bev_distance(left: Any, right: Any) -> float:
    """Compute Euclidean BEV distance for objects or dicts with x/y fields."""
    dx = _get_number(left, "x") - _get_number(right, "x")
    dy = _get_number(left, "y") - _get_number(right, "y")
    return math.hypot(dx, dy)


def mahalanobis_distance(
    left: Any,
    right: Any,
    covariance_xy: Sequence[Sequence[float]] | None,
) -> float:
    """Compute 2D Mahalanobis distance, falling back to BEV distance if singular."""
    if covariance_xy is None:
        return bev_distance(left, right)
    if len(covariance_xy) != 2 or any(len(row) != 2 for row in covariance_xy):
        return bev_distance(left, right)

    a = float(covariance_xy[0][0])
    b = float(covariance_xy[0][1])
    c = float(covariance_xy[1][0])
    d = float(covariance_xy[1][1])
    determinant = a * d - b * c
    if abs(determinant) < 1e-12:
        return bev_distance(left, right)

    dx = _get_number(left, "x") - _get_number(right, "x")
    dy = _get_number(left, "y") - _get_number(right, "y")
    inv00 = d / determinant
    inv01 = -b / determinant
    inv10 = -c / determinant
    inv11 = a / determinant
    squared = dx * (inv00 * dx + inv01 * dy) + dy * (inv10 * dx + inv11 * dy)
    return math.sqrt(max(0.0, squared))


def class_compatible(left_type: str | None, right_type: str | None) -> bool:
    """Return whether two semantic classes can be associated."""
    left = _normalize_class(left_type)
    right = _normalize_class(right_type)
    if left == "unknown" or right == "unknown":
        return True
    return left == right


def gated_association_cost(
    left: Any,
    right: Any,
    *,
    distance_gate_m: float,
    mahalanobis_gate: float,
) -> float | None:
    """Return association cost if BEV or Mahalanobis gating accepts the pair."""
    if not class_compatible(
        _get_optional_string(left, "type"),
        _get_optional_string(right, "type"),
    ):
        return None

    distance = bev_distance(left, right)
    if distance <= distance_gate_m:
        return distance

    covariance = _get_optional_value(left, "covariance_xy") or _get_optional_value(
        right, "covariance_xy"
    )
    maha = mahalanobis_distance(left, right, covariance)
    if maha <= mahalanobis_gate:
        return distance
    return None


def hungarian_assignment(
    cost_matrix: Sequence[Sequence[float | None]],
    *,
    max_cost: float | None = None,
) -> list[AssociationMatch]:
    """Solve a small rectangular assignment problem deterministically.

    This exact enumerator is sufficient for synthetic smoke tests and avoids
    adding SciPy as a default dependency.
    """
    row_count = len(cost_matrix)
    if row_count == 0:
        return []
    column_count = len(cost_matrix[0])
    if column_count == 0:
        return []
    if any(len(row) != column_count for row in cost_matrix):
        raise ValueError("cost_matrix must be rectangular")

    if row_count <= column_count:
        return _assign_rows_to_columns(cost_matrix, max_cost=max_cost)

    transposed = [
        [cost_matrix[row_index][column_index] for row_index in range(row_count)]
        for column_index in range(column_count)
    ]
    transposed_matches = _assign_rows_to_columns(transposed, max_cost=max_cost)
    return sorted(
        [
            AssociationMatch(
                left_index=match.right_index,
                right_index=match.left_index,
                cost=match.cost,
            )
            for match in transposed_matches
        ],
        key=lambda match: (match.left_index, match.right_index, match.cost),
    )


def associate_by_bev(
    left_items: Sequence[Any],
    right_items: Sequence[Any],
    *,
    distance_gate_m: float = 2.0,
    mahalanobis_gate: float = 3.0,
) -> AssociationResult:
    """Associate two object sets with class compatibility and distance gates."""
    cost_matrix: list[list[float | None]] = []
    for left in left_items:
        row: list[float | None] = []
        for right in right_items:
            row.append(
                gated_association_cost(
                    left,
                    right,
                    distance_gate_m=distance_gate_m,
                    mahalanobis_gate=mahalanobis_gate,
                )
            )
        cost_matrix.append(row)

    matches = hungarian_assignment(cost_matrix)
    matched_left = {match.left_index for match in matches}
    matched_right = {match.right_index for match in matches}
    return AssociationResult(
        matches=matches,
        unmatched_left=[index for index in range(len(left_items)) if index not in matched_left],
        unmatched_right=[index for index in range(len(right_items)) if index not in matched_right],
    )


def _assign_rows_to_columns(
    cost_matrix: Sequence[Sequence[float | None]],
    *,
    max_cost: float | None,
) -> list[AssociationMatch]:
    row_count = len(cost_matrix)
    column_count = len(cost_matrix[0])
    best_total: float | None = None
    best_columns: tuple[int, ...] | None = None

    for columns in itertools.permutations(range(column_count), row_count):
        total = 0.0
        valid = True
        for row_index, column_index in enumerate(columns):
            cost = cost_matrix[row_index][column_index]
            if cost is None or not math.isfinite(float(cost)):
                valid = False
                break
            if max_cost is not None and cost > max_cost:
                valid = False
                break
            total += float(cost)
        if not valid:
            continue
        if best_total is None or (total, columns) < (best_total, best_columns):
            best_total = total
            best_columns = columns

    if best_columns is None:
        return []
    return [
        AssociationMatch(
            left_index=row_index,
            right_index=column_index,
            cost=float(cost_matrix[row_index][column_index]),
        )
        for row_index, column_index in enumerate(best_columns)
    ]


def _normalize_class(value: str | None) -> str:
    raw = "" if value is None else value.strip().lower()
    for group_name, members in CLASS_GROUPS.items():
        if raw in members:
            return group_name
    return raw


def _get_number(item: Any, field_name: str) -> float:
    value = _get_optional_value(item, field_name)
    if not isinstance(value, Real):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def _get_optional_string(item: Any, field_name: str) -> str | None:
    value = _get_optional_value(item, field_name)
    return value if isinstance(value, str) else None


def _get_optional_value(item: Any, field_name: str) -> Any:
    if isinstance(item, dict):
        return item.get(field_name)
    return getattr(item, field_name, None)
