"""Calibration interfaces for CoordiWorld risk probabilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from coordiworld.risks.geometry import clip01


@dataclass(frozen=True)
class BinningCalibrator:
    method: str
    bin_edges: list[float]
    bin_values: list[float]
    default_value: float

    def apply(self, score: float) -> float:
        if not self.bin_edges:
            return clip01(self.default_value)
        for edge, value in zip(self.bin_edges, self.bin_values):
            if score <= edge:
                return clip01(value)
        return clip01(self.bin_values[-1])

    def apply_many(self, scores: Sequence[float]) -> list[float]:
        return [self.apply(score) for score in scores]


def fit_calibrator(
    scores: Sequence[float],
    labels: Sequence[bool | int | float],
    *,
    n_bins: int = 10,
    method: str = "isotonic",
) -> BinningCalibrator:
    """Fit isotonic-style binned calibration with a binning fallback."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("scores must not be empty")
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    if method not in {"isotonic", "binning"}:
        raise ValueError("method must be 'isotonic' or 'binning'")

    pairs = sorted((float(score), float(label)) for score, label in zip(scores, labels))
    bin_count = min(n_bins, len(pairs))
    bins: list[list[tuple[float, float]]] = [[] for _ in range(bin_count)]
    for index, pair in enumerate(pairs):
        bins[min(index * bin_count // len(pairs), bin_count - 1)].append(pair)

    edges = [max(score for score, _ in bucket) for bucket in bins]
    values = [sum(label for _, label in bucket) / len(bucket) for bucket in bins]
    weights = [float(len(bucket)) for bucket in bins]
    if method == "isotonic":
        values = _pava(values, weights)

    return BinningCalibrator(
        method=method,
        bin_edges=edges,
        bin_values=[clip01(value) for value in values],
        default_value=clip01(sum(float(label) for label in labels) / len(labels)),
    )


def save_calibrator(calibrator: BinningCalibrator, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(asdict(calibrator), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_calibrator(path: str | Path) -> BinningCalibrator:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return BinningCalibrator(
        method=data["method"],
        bin_edges=[float(value) for value in data["bin_edges"]],
        bin_values=[float(value) for value in data["bin_values"]],
        default_value=float(data["default_value"]),
    )


def _pava(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    levels: list[float] = []
    level_weights: list[float] = []
    counts: list[int] = []
    for value, weight in zip(values, weights):
        levels.append(float(value))
        level_weights.append(float(weight))
        counts.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            merged_weight = level_weights[-2] + level_weights[-1]
            merged_level = (
                levels[-2] * level_weights[-2] + levels[-1] * level_weights[-1]
            ) / merged_weight
            merged_count = counts[-2] + counts[-1]
            levels[-2:] = [merged_level]
            level_weights[-2:] = [merged_weight]
            counts[-2:] = [merged_count]

    expanded: list[float] = []
    for level, count in zip(levels, counts):
        expanded.extend([level] * count)
    return expanded
