"""Calibration metrics for synthetic evaluator diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    confidence: float
    accuracy: float
    gap: float


@dataclass(frozen=True)
class CalibrationMetrics:
    ece: float
    brier_score: float
    bins: list[ReliabilityBin]


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[bool | int | float],
    *,
    n_bins: int = 10,
) -> float:
    """Compute ECE with fixed-width bins over [0, 1]."""
    bins = reliability_bins(probabilities, labels, n_bins=n_bins)
    total = sum(bucket.count for bucket in bins)
    if total == 0:
        return 0.0
    return sum(bucket.count / total * bucket.gap for bucket in bins)


def brier_score(
    probabilities: Sequence[float],
    labels: Sequence[bool | int | float],
) -> float:
    """Compute binary Brier score."""
    _validate_probabilities_and_labels(probabilities, labels)
    return sum(
        (float(probability) - float(label)) ** 2
        for probability, label in zip(probabilities, labels)
    ) / len(probabilities)


def reliability_bins(
    probabilities: Sequence[float],
    labels: Sequence[bool | int | float],
    *,
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Return reliability-bin summaries for diagnostics."""
    _validate_probabilities_and_labels(probabilities, labels)
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for probability, label in zip(probabilities, labels):
        index = min(int(float(probability) * n_bins), n_bins - 1)
        buckets[index].append((float(probability), float(label)))

    result: list[ReliabilityBin] = []
    for index, bucket in enumerate(buckets):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if bucket:
            confidence = sum(probability for probability, _ in bucket) / len(bucket)
            accuracy = sum(label for _, label in bucket) / len(bucket)
        else:
            confidence = 0.0
            accuracy = 0.0
        result.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=len(bucket),
                confidence=confidence,
                accuracy=accuracy,
                gap=abs(confidence - accuracy),
            )
        )
    return result


def compute_calibration_metrics(
    probabilities: Sequence[float],
    labels: Sequence[bool | int | float],
    *,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute ECE, Brier score, and reliability-bin diagnostics."""
    bins = reliability_bins(probabilities, labels, n_bins=n_bins)
    total = sum(bucket.count for bucket in bins)
    ece = 0.0 if total == 0 else sum(bucket.count / total * bucket.gap for bucket in bins)
    return CalibrationMetrics(
        ece=ece,
        brier_score=brier_score(probabilities, labels),
        bins=bins,
    )


def _validate_probabilities_and_labels(
    probabilities: Sequence[float],
    labels: Sequence[bool | int | float],
) -> None:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have equal length")
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    for probability in probabilities:
        value = float(probability)
        if value < 0.0 or value > 1.0:
            raise ValueError("probabilities must be in [0, 1]")
    for label in labels:
        value = float(label)
        if value not in {0.0, 1.0}:
            raise ValueError("labels must be binary")
