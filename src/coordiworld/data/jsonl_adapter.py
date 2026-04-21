"""Standardized JSONL adapter for converted CoordiWorld scenario samples."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coordiworld.data.base import (
    BaseScenarioSample,
    DatasetFormatError,
    ScenarioLabels,
    validate_base_scenario_sample,
)
from coordiworld.scene_summary.io import scene_summary_from_dict, scene_summary_to_dict
from coordiworld.scene_summary.validators import validate_scene_summary


@dataclass(frozen=True)
class JsonlDatasetConfig:
    path: str | Path
    split: str = "val"
    max_samples: int | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "JsonlDatasetConfig":
        mapping = data or {}
        path = mapping.get("path") or mapping.get("file")
        if path is None:
            raise DatasetFormatError("jsonl config requires path")
        return cls(
            path=Path(str(path)),
            split=str(mapping.get("split", cls.split)),
            max_samples=_optional_int(mapping.get("max_samples")),
        )


class JsonlScenarioDataset:
    """Read standardized BaseScenarioSample records from JSONL."""

    def __init__(self, config: JsonlDatasetConfig) -> None:
        self.config = config
        self.path = Path(config.path)
        if not self.path.exists():
            raise DatasetFormatError(f"JSONL dataset file does not exist: {self.path}")
        if not self.path.is_file():
            raise DatasetFormatError(f"JSONL dataset path is not a file: {self.path}")
        self._samples = self._load_samples()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> BaseScenarioSample:
        return self._samples[index]

    def iter_samples(self, split: str | None = None) -> Iterator[BaseScenarioSample]:
        resolved_split = split or self.config.split
        for sample in self._samples:
            sample_split = sample.metadata.get("split")
            if sample_split is None or sample_split == resolved_split:
                yield sample

    def _load_samples(self) -> list[BaseScenarioSample]:
        samples: list[BaseScenarioSample] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                samples.append(self._parse_line(line, line_number))
                if self.config.max_samples is not None and len(samples) >= self.config.max_samples:
                    break
        if not samples:
            raise DatasetFormatError(f"JSONL dataset contains no samples: {self.path}")
        return samples

    def _parse_line(self, line: str, line_number: int) -> BaseScenarioSample:
        try:
            data = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetFormatError(
                f"{self.path}:{line_number} is not valid JSON: {error}"
            ) from error
        if not isinstance(data, dict):
            raise DatasetFormatError(f"{self.path}:{line_number} must be a JSON object")
        try:
            sample = scenario_sample_from_dict(data)
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetFormatError(
                f"{self.path}:{line_number} invalid sample: {error}"
            ) from error
        return sample


def scenario_sample_from_dict(data: dict[str, Any]) -> BaseScenarioSample:
    """Parse one standardized scenario-sample dict."""
    mapping = _require_mapping(data, "scenario sample")
    _require_fields(
        mapping,
        [
            "scene_id",
            "timestamp",
            "scene_summary_history",
            "candidate_trajectories",
            "logged_ego_future",
            "labels",
        ],
    )
    history = [
        scene_summary_from_dict(summary)
        for summary in _require_list(mapping["scene_summary_history"], "scene_summary_history")
    ]
    for index, summary in enumerate(history):
        try:
            validate_scene_summary(summary)
        except ValueError as error:
            raise DatasetFormatError(f"scene_summary_history[{index}] invalid: {error}") from error

    labels = labels_from_dict(_require_mapping(mapping["labels"], "labels"))
    sample = BaseScenarioSample(
        scene_id=str(mapping["scene_id"]),
        sample_id=str(mapping.get("sample_id", f"{mapping['scene_id']}-sample-000")),
        timestamp=float(mapping["timestamp"]),
        coordinate_frame=str(mapping.get("coordinate_frame", history[-1].coordinate_frame)),
        scene_summary_history=history,
        candidate_trajectories=_trajectory_list(
            mapping["candidate_trajectories"], "candidate_trajectories"
        ),
        logged_ego_future=_trajectory(mapping["logged_ego_future"], "logged_ego_future"),
        future_agents=_future_agents(mapping.get("future_agents")),
        labels=labels,
        transform_metadata=dict(mapping.get("transform_metadata", {})),
        provenance=dict(mapping.get("provenance", {})),
        quality_flags=list(mapping.get("quality_flags", [])),
        metadata=dict(mapping.get("metadata", {})),
    )
    validate_base_scenario_sample(sample)
    return sample


def scenario_sample_to_dict(sample: BaseScenarioSample) -> dict[str, Any]:
    """Serialize one BaseScenarioSample to standardized JSON-compatible dict."""
    validate_base_scenario_sample(sample)
    return {
        "scene_id": sample.scene_id,
        "sample_id": sample.sample_id,
        "timestamp": sample.timestamp,
        "coordinate_frame": sample.coordinate_frame,
        "scene_summary_history": [
            scene_summary_to_dict(summary) for summary in sample.scene_summary_history
        ],
        "candidate_trajectories": sample.candidate_trajectories,
        "logged_ego_future": sample.logged_ego_future,
        "future_agents": sample.future_agents,
        "labels": {
            "collision": sample.labels.collision,
            "violation": sample.labels.violation,
            "pseudo_sim_score": sample.labels.pseudo_sim_score,
            "progress": sample.labels.progress,
        },
        "transform_metadata": sample.transform_metadata,
        "provenance": sample.provenance,
        "quality_flags": sample.quality_flags,
        "metadata": sample.metadata,
    }


def labels_from_dict(data: dict[str, Any]) -> ScenarioLabels:
    _require_fields(data, ["collision", "violation", "pseudo_sim_score", "progress"])
    return ScenarioLabels(
        collision=bool(data["collision"]),
        violation=bool(data["violation"]),
        pseudo_sim_score=float(data["pseudo_sim_score"]),
        progress=float(data["progress"]),
    )


def build_jsonl_dataset(config: dict[str, Any] | None = None) -> JsonlScenarioDataset:
    return JsonlScenarioDataset(JsonlDatasetConfig.from_mapping(config))


def _require_fields(mapping: dict[str, Any], fields: list[str]) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise DatasetFormatError(f"missing required fields: {missing}")


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetFormatError(f"{path} must be a dict")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise DatasetFormatError(f"{path} must be a list")
    return value


def _trajectory(value: Any, path: str) -> list[list[float]]:
    poses = _require_list(value, path)
    trajectory: list[list[float]] = []
    for index, pose in enumerate(poses):
        if not isinstance(pose, list) or len(pose) != 3:
            raise DatasetFormatError(f"{path}[{index}] must be [x, y, yaw]")
        trajectory.append([float(pose[0]), float(pose[1]), float(pose[2])])
    return trajectory


def _trajectory_list(value: Any, path: str) -> list[list[list[float]]]:
    trajectories = _require_list(value, path)
    return [
        _trajectory(trajectory, f"{path}[{index}]")
        for index, trajectory in enumerate(trajectories)
    ]


def _future_agents(value: Any) -> dict[str, list[list[float]]] | None:
    if value is None:
        return None
    mapping = _require_mapping(value, "future_agents")
    return {
        str(agent_id): _trajectory(trajectory, f"future_agents[{agent_id!r}]")
        for agent_id, trajectory in mapping.items()
    }


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
