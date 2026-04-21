"""Tests for dataset registry and adapter construction."""

from __future__ import annotations

from coordiworld.data.registry import available_datasets, build_dataset, get_dataset_adapter
from coordiworld.data.synthetic import SyntheticScenarioDataset


def test_registry_lists_required_dataset_names() -> None:
    assert set(available_datasets()) >= {
        "synthetic",
        "jsonl",
        "navsim",
        "openscene",
        "nuscenes",
        "waymo",
    }


def test_registry_builds_synthetic_dataset() -> None:
    dataset = build_dataset("synthetic", {"num_samples": 1, "history_length": 1})

    assert isinstance(dataset, SyntheticScenarioDataset)
    assert len(dataset) == 1


def test_registry_returns_builder() -> None:
    builder = get_dataset_adapter("synthetic")

    assert callable(builder)
