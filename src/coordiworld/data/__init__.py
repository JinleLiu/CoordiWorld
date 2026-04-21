"""Dataset contracts, adapters, and synthetic fixtures for CoordiWorld."""

from coordiworld.data.base import (
    BaseScenarioSample,
    CandidateTrajectories,
    DataRootError,
    DatasetAdapter,
    DatasetFormatError,
    DatasetSplit,
    FutureAgentState,
    MissingDependencyError,
    ScenarioDataset,
    ScenarioLabels,
    Trajectory,
    candidate_pool_shape,
    validate_base_scenario_sample,
)
from coordiworld.data.candidate_pool import (
    CandidatePool,
    CandidatePoolConfig,
    build_candidate_pool,
    build_shared_candidate_pool,
)
from coordiworld.data.jsonl_adapter import JsonlDatasetConfig, JsonlScenarioDataset
from coordiworld.data.registry import (
    available_datasets,
    build_dataset,
    get_dataset_adapter,
    register_dataset,
)
from coordiworld.data.synthetic import SyntheticDatasetConfig, SyntheticScenarioDataset

__all__ = [
    "BaseScenarioSample",
    "CandidatePool",
    "CandidatePoolConfig",
    "CandidateTrajectories",
    "DataRootError",
    "DatasetAdapter",
    "DatasetFormatError",
    "DatasetSplit",
    "FutureAgentState",
    "JsonlDatasetConfig",
    "JsonlScenarioDataset",
    "MissingDependencyError",
    "ScenarioDataset",
    "ScenarioLabels",
    "SyntheticDatasetConfig",
    "SyntheticScenarioDataset",
    "Trajectory",
    "available_datasets",
    "build_candidate_pool",
    "build_dataset",
    "build_shared_candidate_pool",
    "candidate_pool_shape",
    "get_dataset_adapter",
    "register_dataset",
    "validate_base_scenario_sample",
]
