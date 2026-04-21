"""Tests for synthetic data adapter samples."""

from __future__ import annotations

import pytest

from coordiworld.data.base import DataRootError, candidate_pool_shape, validate_base_scenario_sample
from coordiworld.data.candidate_pool import CandidatePoolConfig
from coordiworld.data.collate import collate_scenario_samples
from coordiworld.data.navsim_adapter import NavsimAdapter
from coordiworld.data.synthetic import SyntheticDatasetConfig, SyntheticScenarioDataset


def test_synthetic_dataset_returns_valid_base_sample() -> None:
    dataset = SyntheticScenarioDataset(
        SyntheticDatasetConfig(
            num_samples=2,
            history_length=3,
            candidate_pool_config=CandidatePoolConfig(horizon_steps=5),
        )
    )

    sample = dataset[0]

    validate_base_scenario_sample(sample)
    assert sample.scene_id == "synthetic-scene-000"
    assert len(sample.scene_summary_history) == 3
    assert candidate_pool_shape(sample.candidate_trajectories) == (7, 5, 3)
    assert len(sample.logged_ego_future) == 5
    assert sample.labels.collision is False
    assert sample.labels.violation is False
    assert sample.metadata["candidate_pool"]["candidate_pool_type"] == "shared"


def test_synthetic_dataset_uses_shared_candidate_pool_across_samples() -> None:
    dataset = SyntheticScenarioDataset(
        SyntheticDatasetConfig(
            num_samples=2,
            candidate_pool_config=CandidatePoolConfig(horizon_steps=4, seed=7),
        )
    )

    first = dataset[0]
    second = dataset[1]

    assert first.candidate_trajectories == second.candidate_trajectories
    assert first.metadata["candidate_pool"]["seed"] == 7
    assert second.metadata["candidate_pool"]["seed"] == 7


def test_synthetic_dataset_iter_samples_split_contract() -> None:
    dataset = SyntheticScenarioDataset(SyntheticDatasetConfig(num_samples=2))

    assert len(list(dataset.iter_samples("synthetic"))) == 2
    assert len(list(dataset.iter_samples("train"))) == 2
    with pytest.raises(ValueError, match="split='synthetic'"):
        list(dataset.iter_samples("unknown"))


def test_collate_preserves_candidate_pool_shape() -> None:
    dataset = SyntheticScenarioDataset(SyntheticDatasetConfig(num_samples=2))
    batch = collate_scenario_samples([dataset[0], dataset[1]])

    assert batch["scene_ids"] == ["synthetic-scene-000", "synthetic-scene-001"]
    assert batch["candidate_pool_shape"] == (7, 6, 3)
    assert batch["labels"]["collision"] == [False, False]


def test_navsim_adapter_requires_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NAVSIM_ROOT", raising=False)

    with pytest.raises(DataRootError, match="NAVSIM root is required"):
        NavsimAdapter().validate_root()


def test_navsim_adapter_is_explicit_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NAVSIM_ROOT", str(__import__("pathlib").Path("/tmp")))

    with pytest.raises(Exception, match="NAVSIM|navsim|Native"):
        list(NavsimAdapter().iter_samples("mini"))
