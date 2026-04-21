"""Synthetic dataset fixtures for CoordiWorld data adapter smoke tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from coordiworld.data.base import BaseScenarioSample, ScenarioLabels, validate_base_scenario_sample
from coordiworld.data.candidate_pool import (
    CandidatePoolConfig,
    build_shared_candidate_pool,
    candidate_pool_config_from_mapping,
)
from coordiworld.scene_summary.schema import AgentState, EgoState, MapToken, SceneSummary


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    num_samples: int = 3
    history_length: int = 2
    candidate_pool_config: CandidatePoolConfig = field(default_factory=CandidatePoolConfig)
    coordinate_frame: str = "ego"
    scene_id_prefix: str = "synthetic-scene"

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "SyntheticDatasetConfig":
        mapping = data or {}
        candidate_pool_data = mapping.get("candidate_pool")
        if candidate_pool_data is not None and not isinstance(candidate_pool_data, dict):
            candidate_pool_data = {}
        num_samples = int(mapping.get("num_samples", mapping.get("max_samples", cls.num_samples)))
        return cls(
            num_samples=num_samples,
            history_length=int(mapping.get("history_length", cls.history_length)),
            candidate_pool_config=candidate_pool_config_from_mapping(candidate_pool_data),
            coordinate_frame=str(mapping.get("coordinate_frame", cls.coordinate_frame)),
            scene_id_prefix=str(mapping.get("scene_id_prefix", cls.scene_id_prefix)),
        )


class SyntheticScenarioDataset:
    """Small deterministic dataset that never reads real benchmark files."""

    def __init__(self, config: SyntheticDatasetConfig | None = None) -> None:
        self.config = config or SyntheticDatasetConfig()
        if self.config.num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        if self.config.history_length <= 0:
            raise ValueError("history_length must be > 0")
        self._candidate_pool = build_shared_candidate_pool(self.config.candidate_pool_config)

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> BaseScenarioSample:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        scene_id = f"{self.config.scene_id_prefix}-{index:03d}"
        history = [
            make_synthetic_scene_summary(
                scene_id=scene_id,
                timestamp=float(index) + history_index * 0.1,
                coordinate_frame=self.config.coordinate_frame,
                sample_index=index,
                history_index=history_index,
            )
            for history_index in range(self.config.history_length)
        ]
        selected_candidate = self._candidate_pool.trajectories[
            index % len(self._candidate_pool.trajectories)
        ]
        logged_future = [[pose[0], pose[1], pose[2]] for pose in selected_candidate]
        sample = BaseScenarioSample(
            scene_id=scene_id,
            sample_id=f"{scene_id}-sample-000",
            timestamp=history[-1].timestamp,
            coordinate_frame=self.config.coordinate_frame,
            scene_summary_history=history,
            candidate_trajectories=self._candidate_pool.trajectories,
            logged_ego_future=logged_future,
            future_agents={
                "agent-0": [
                    [step[0] + 8.0, step[1] + 1.5, step[2]]
                    for step in logged_future
                ]
            },
            labels=ScenarioLabels(
                collision=False,
                violation=False,
                pseudo_sim_score=0.8 + 0.01 * index,
                progress=logged_future[-1][0],
            ),
            provenance={"dataset": "synthetic", "real_data": False},
            quality_flags=["synthetic_fixture"],
            metadata={"candidate_pool": self._candidate_pool.metadata},
        )
        validate_base_scenario_sample(sample)
        return sample

    def __iter__(self) -> Iterator[BaseScenarioSample]:
        for index in range(len(self)):
            yield self[index]

    def iter_samples(self, split: str = "synthetic") -> Iterator[BaseScenarioSample]:
        if split not in {"synthetic", "train", "val", "test", "mini"}:
            raise ValueError("SyntheticScenarioDataset only supports split='synthetic'")
        yield from self


def make_synthetic_scene_summary(
    *,
    scene_id: str,
    timestamp: float,
    coordinate_frame: str,
    sample_index: int,
    history_index: int,
) -> SceneSummary:
    ego_x = float(sample_index) + history_index * 0.5
    ego = EgoState(
        x=ego_x,
        y=0.0,
        yaw=0.0,
        vx=5.0,
        vy=0.0,
        length=4.8,
        width=2.0,
    )
    agent = AgentState(
        id="agent-0",
        type="vehicle",
        x=ego_x + 8.0,
        y=1.5,
        yaw=0.0,
        vx=4.5,
        vy=0.0,
        length=4.5,
        width=1.9,
        confidence=0.95,
        covariance_xy=[[0.4, 0.0], [0.0, 0.4]],
        existence_prob=0.99,
        source_modalities=["synthetic"],
        source_ids=[f"synthetic-agent-{sample_index}"],
        fusion_lineage=["synthetic_fixture"],
        ambiguity_flags=[],
        semantic_attributes={"fixture": True},
    )
    lane = MapToken(
        id="lane-0",
        type="lane_centerline",
        polyline=[[ego_x, 0.0], [ego_x + 20.0, 0.0]],
        polygon=None,
        traffic_state=None,
        rule_attributes={"synthetic": True},
    )
    return SceneSummary(
        scene_id=scene_id,
        timestamp=timestamp,
        coordinate_frame=coordinate_frame,
        ego=ego,
        agents=[agent],
        map_tokens=[lane],
        provenance={"dataset": "synthetic"},
        metadata={"sample_index": sample_index, "history_index": history_index},
    )
