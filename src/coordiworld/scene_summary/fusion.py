"""Deterministic fusion primitives for ICA-style SceneSummary generation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class FusionTrace:
    source_modalities: list[str]
    source_ids: list[str]
    fusion_lineage: list[str]
    ambiguity_flags: list[str]


def continuous_weighted_fusion(values: Sequence[float], weights: Sequence[float]) -> float:
    """Fuse continuous scalar values with deterministic weighted averaging."""
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if not values:
        raise ValueError("values must not be empty")

    weighted_sum = 0.0
    weight_sum = 0.0
    for value, weight in zip(values, weights):
        if not isinstance(value, Real) or not isinstance(weight, Real):
            raise ValueError("values and weights must be numeric")
        if weight < 0:
            raise ValueError("weights must be non-negative")
        weighted_sum += float(value) * float(weight)
        weight_sum += float(weight)

    if weight_sum == 0:
        return sum(float(value) for value in values) / len(values)
    return weighted_sum / weight_sum


def categorical_weighted_vote(labels: Sequence[str | None], weights: Sequence[float]) -> str:
    """Choose a categorical label by total weight, then lexicographic tie-break."""
    if len(labels) != len(weights):
        raise ValueError("labels and weights must have the same length")
    totals: dict[str, float] = defaultdict(float)
    for label, weight in zip(labels, weights):
        key = "unknown" if label is None or not label.strip() else label.strip()
        totals[key] += float(weight)
    if not totals:
        return "unknown"
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))[0][0]


def merge_semantic_attributes(items: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Merge semantic attributes without letting later camera facts affect geometry."""
    merged: dict[str, object] = {}
    for attributes in items:
        for key in sorted(attributes):
            value = attributes[key]
            if key not in merged:
                merged[key] = value
            elif merged[key] != value:
                merged[f"{key}_candidates"] = sorted(
                    {str(merged[key]), str(value)}
                )
    return merged


def generate_ambiguity_flags(
    *,
    type_votes: Sequence[str | None],
    semantic_conflicts: Mapping[str, Sequence[object]] | None = None,
) -> list[str]:
    """Generate ambiguity flags for conflicting but resolved fused facts."""
    flags: set[str] = set()
    normalized_types = {
        label.strip()
        for label in type_votes
        if label is not None and label.strip()
    }
    if len(normalized_types) > 1:
        flags.add("class_conflict")
        flags.add("conflict_resolved")

    if semantic_conflicts:
        for key, values in semantic_conflicts.items():
            distinct_values = {str(value) for value in values}
            if len(distinct_values) > 1:
                flags.add(f"{key}_conflict")
                flags.add("conflict_resolved")

    return sorted(flags)


def build_fusion_trace(
    facts: Iterable[Any],
    *,
    stages: Sequence[str],
    ambiguity_flags: Sequence[str] = (),
) -> FusionTrace:
    """Build deterministic source/provenance fields for an AgentState."""
    modalities: set[str] = set()
    source_ids: set[str] = set()
    for fact in facts:
        modality = getattr(fact, "modality", None)
        source_id = getattr(fact, "source_id", None)
        if isinstance(modality, str) and modality:
            modalities.add(modality)
        if isinstance(source_id, str) and source_id:
            source_ids.add(source_id)

    return FusionTrace(
        source_modalities=sorted(modalities),
        source_ids=sorted(source_ids),
        fusion_lineage=list(dict.fromkeys(stages)),
        ambiguity_flags=sorted(set(ambiguity_flags)),
    )


def collect_semantic_conflicts(facts: Iterable[Any]) -> dict[str, list[object]]:
    """Collect multi-valued semantic attributes for ambiguity flag generation."""
    values_by_key: dict[str, list[object]] = defaultdict(list)
    for fact in facts:
        attributes = getattr(fact, "semantic_attributes", None)
        if not isinstance(attributes, dict):
            continue
        for key, value in attributes.items():
            values_by_key[key].append(value)
    return {
        key: values
        for key, values in values_by_key.items()
        if len({str(value) for value in values}) > 1
    }
