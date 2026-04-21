"""Tests for standardized JSONL scenario sample adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from coordiworld.data.base import DatasetFormatError, validate_base_scenario_sample
from coordiworld.data.jsonl_adapter import JsonlDatasetConfig, JsonlScenarioDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_JSONL = REPO_ROOT / "examples" / "data" / "scenario_sample_minimal.jsonl"


def test_jsonl_example_loads_valid_sample() -> None:
    dataset = JsonlScenarioDataset(JsonlDatasetConfig(path=EXAMPLE_JSONL, split="val"))

    sample = dataset[0]

    validate_base_scenario_sample(sample)
    assert len(dataset) == 1
    assert sample.scene_id == "example-scene-000"
    assert len(sample.candidate_trajectories) == 2


def test_jsonl_iter_samples_respects_split() -> None:
    dataset = JsonlScenarioDataset(JsonlDatasetConfig(path=EXAMPLE_JSONL, split="val"))

    assert len(list(dataset.iter_samples("val"))) == 1
    assert len(list(dataset.iter_samples("train"))) == 0


def test_jsonl_missing_required_field_raises_clear_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text('{"scene_id": "bad"}\n', encoding="utf-8")

    with pytest.raises(DatasetFormatError, match="missing required fields"):
        JsonlScenarioDataset(JsonlDatasetConfig(path=bad_path))
