"""Dataset adapter registry for CoordiWorld."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from coordiworld.data.base import ScenarioDataset
from coordiworld.data.jsonl_adapter import build_jsonl_dataset
from coordiworld.data.navsim_adapter import build_navsim_dataset
from coordiworld.data.nuscenes_adapter import build_nuscenes_dataset
from coordiworld.data.openscene_adapter import build_openscene_dataset
from coordiworld.data.synthetic import SyntheticDatasetConfig, SyntheticScenarioDataset
from coordiworld.data.waymo_adapter import build_waymo_dataset

DatasetBuilder = Callable[[dict[str, Any] | None], ScenarioDataset]

_DATASET_REGISTRY: dict[str, DatasetBuilder] = {}


def register_dataset(name: str, builder: DatasetBuilder) -> None:
    """Register or replace a dataset builder."""
    normalized = _normalize_name(name)
    if not callable(builder):
        raise TypeError("builder must be callable")
    _DATASET_REGISTRY[normalized] = builder


def available_datasets() -> tuple[str, ...]:
    """Return registered dataset names."""
    _ensure_defaults_registered()
    return tuple(sorted(_DATASET_REGISTRY))


def get_dataset_adapter(name: str) -> DatasetBuilder:
    """Return a registered dataset builder."""
    _ensure_defaults_registered()
    normalized = _normalize_name(name)
    try:
        return _DATASET_REGISTRY[normalized]
    except KeyError as error:
        raise KeyError(
            f"Unknown dataset {name!r}. Available datasets: {', '.join(available_datasets())}"
        ) from error


def build_dataset(name: str, config: dict[str, Any] | None = None) -> ScenarioDataset:
    """Build a dataset adapter by name."""
    return get_dataset_adapter(name)(config)


def _build_synthetic_dataset(config: dict[str, Any] | None = None) -> SyntheticScenarioDataset:
    return SyntheticScenarioDataset(SyntheticDatasetConfig.from_mapping(config))


def _ensure_defaults_registered() -> None:
    if _DATASET_REGISTRY:
        return
    register_dataset("synthetic", _build_synthetic_dataset)
    register_dataset("jsonl", build_jsonl_dataset)
    register_dataset("navsim", build_navsim_dataset)
    register_dataset("openscene", build_openscene_dataset)
    register_dataset("nuscenes", build_nuscenes_dataset)
    register_dataset("waymo", build_waymo_dataset)


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("dataset name must be non-empty")
    return normalized
