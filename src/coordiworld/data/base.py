"""Canonical data sample contracts for CoordiWorld evaluator inputs."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Protocol

from coordiworld.scene_summary.schema import SceneSummary

Trajectory = list[list[float]]
CandidateTrajectories = list[Trajectory]
DatasetSplit = Literal["train", "val", "test", "mini", "synthetic"]


class DataRootError(RuntimeError):
    """Raised when a required dataset root is missing or inaccessible."""


class MissingDependencyError(RuntimeError):
    """Raised when an optional official dataset dependency is unavailable."""


class DatasetFormatError(ValueError):
    """Raised when an adapter cannot parse a dataset record."""


@dataclass(frozen=True)
class ScenarioLabels:
    collision: bool
    violation: bool
    pseudo_sim_score: float
    progress: float


@dataclass(frozen=True)
class FutureAgentState:
    """Optional typed future-agent trajectory record for future adapters."""

    agent_id: str
    trajectory: Trajectory
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BaseScenarioSample:
    scene_id: str
    sample_id: str
    timestamp: float
    coordinate_frame: str
    scene_summary_history: list[SceneSummary]
    candidate_trajectories: CandidateTrajectories
    logged_ego_future: Trajectory
    future_agents: dict[str, Trajectory] | None
    labels: ScenarioLabels
    transform_metadata: dict[str, object] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_trajectories)

    @property
    def horizon_steps(self) -> int:
        return len(self.logged_ego_future)


class ScenarioDataset(Protocol):
    """Protocol shared by synthetic, JSONL, and real dataset adapters."""

    def __len__(self) -> int:
        """Return available sample count if known."""

    def __getitem__(self, index: int) -> BaseScenarioSample:
        """Return one sample by index."""

    def iter_samples(self, split: str = "val") -> Any:
        """Yield samples for a split."""


DatasetAdapter = ScenarioDataset


def validate_base_scenario_sample(sample: BaseScenarioSample) -> None:
    """Validate the shape and scalar fields of a BaseScenarioSample."""
    if not isinstance(sample, BaseScenarioSample):
        raise ValueError("sample must be a BaseScenarioSample")
    if not _is_nonempty_string(sample.scene_id):
        raise ValueError("scene_id must be a non-empty string")
    if not _is_nonempty_string(sample.sample_id):
        raise ValueError("sample_id must be a non-empty string")
    if not _is_nonempty_string(sample.coordinate_frame):
        raise ValueError("coordinate_frame must be a non-empty string")
    _require_finite_number(sample.timestamp, "timestamp")

    if not isinstance(sample.scene_summary_history, list) or not sample.scene_summary_history:
        raise ValueError("scene_summary_history must be a non-empty list")
    for index, summary in enumerate(sample.scene_summary_history):
        if not isinstance(summary, SceneSummary):
            raise ValueError(f"scene_summary_history[{index}] must be a SceneSummary")

    horizon = _validate_trajectory(sample.logged_ego_future, "logged_ego_future")
    _validate_candidate_trajectories(sample.candidate_trajectories, horizon)
    if sample.future_agents is not None:
        if not isinstance(sample.future_agents, dict):
            raise ValueError("future_agents must be a dict or None")
        for agent_id, trajectory in sample.future_agents.items():
            if not _is_nonempty_string(agent_id):
                raise ValueError("future_agents keys must be non-empty strings")
            agent_horizon = _validate_trajectory(trajectory, f"future_agents[{agent_id!r}]")
            if agent_horizon != horizon:
                raise ValueError(f"future_agents[{agent_id!r}] must have horizon {horizon}")

    if not isinstance(sample.labels, ScenarioLabels):
        raise ValueError("labels must be a ScenarioLabels")
    _require_finite_number(sample.labels.pseudo_sim_score, "labels.pseudo_sim_score")
    _require_finite_number(sample.labels.progress, "labels.progress")
    if not isinstance(sample.labels.collision, bool):
        raise ValueError("labels.collision must be bool")
    if not isinstance(sample.labels.violation, bool):
        raise ValueError("labels.violation must be bool")

    _require_dict(sample.transform_metadata, "transform_metadata")
    _require_dict(sample.provenance, "provenance")
    if not isinstance(sample.quality_flags, list):
        raise ValueError("quality_flags must be a list")
    _require_dict(sample.metadata, "metadata")


def candidate_pool_shape(candidate_trajectories: CandidateTrajectories) -> tuple[int, int, int]:
    """Return the [M, H, 3] shape of a candidate trajectory pool."""
    _validate_candidate_trajectories(candidate_trajectories, expected_horizon=None)
    horizon = len(candidate_trajectories[0])
    return len(candidate_trajectories), horizon, 3


def validate_future_agent_state(state: FutureAgentState) -> None:
    """Validate a typed FutureAgentState object."""
    if not isinstance(state, FutureAgentState):
        raise ValueError("state must be a FutureAgentState")
    if not _is_nonempty_string(state.agent_id):
        raise ValueError("agent_id must be a non-empty string")
    _validate_trajectory(state.trajectory, "trajectory")
    _require_dict(state.metadata, "metadata")


def resolve_data_root(
    *,
    explicit_root: str | Path | None,
    env_var: str,
    dataset_name: str,
) -> Path:
    """Resolve a dataset root from config/root argument or an environment variable."""
    raw_root = _resolve_root_value(explicit_root, env_var)
    if not raw_root:
        raise DataRootError(
            f"{dataset_name} root is required. Provide config.root/--root or set {env_var}."
        )
    root = Path(raw_root).expanduser()
    if not root.exists():
        raise DataRootError(f"{dataset_name} root does not exist: {root}")
    if not root.is_dir():
        raise DataRootError(f"{dataset_name} root is not a directory: {root}")
    return root


def _validate_candidate_trajectories(
    trajectories: CandidateTrajectories,
    expected_horizon: int | None,
) -> None:
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("candidate_trajectories must be a non-empty [M,H,3] list")
    first_horizon = _validate_trajectory(trajectories[0], "candidate_trajectories[0]")
    horizon = first_horizon if expected_horizon is None else expected_horizon
    if first_horizon != horizon:
        raise ValueError(f"candidate_trajectories[0] must have horizon {horizon}")
    for index, trajectory in enumerate(trajectories[1:], start=1):
        trajectory_horizon = _validate_trajectory(trajectory, f"candidate_trajectories[{index}]")
        if trajectory_horizon != horizon:
            raise ValueError(f"candidate_trajectories[{index}] must have horizon {horizon}")


def _validate_trajectory(trajectory: Any, path: str) -> int:
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"{path} must be a non-empty [H,3] list")
    for step_index, pose in enumerate(trajectory):
        if not isinstance(pose, list) or len(pose) != 3:
            raise ValueError(f"{path}[{step_index}] must be [x, y, yaw]")
        for value_index, value in enumerate(pose):
            _require_finite_number(value, f"{path}[{step_index}][{value_index}]")
    return len(trajectory)


def _require_finite_number(value: Any, path: str) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{path} must be a finite number")


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_dict(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a dict")


def _resolve_root_value(explicit_root: str | Path | None, env_var: str) -> str:
    if explicit_root is not None:
        raw_value = str(explicit_root).strip()
        if raw_value.startswith("${") and raw_value.endswith("}"):
            return os.environ.get(raw_value[2:-1], "").strip()
        return raw_value
    return os.environ.get(env_var, "").strip()
