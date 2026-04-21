"""Collation helpers for BaseScenarioSample batches without tensor dependencies."""

from __future__ import annotations

from collections.abc import Sequence

from coordiworld.data.base import BaseScenarioSample, candidate_pool_shape


def collate_scenario_samples(samples: Sequence[BaseScenarioSample]) -> dict[str, object]:
    """Collate samples into nested Python lists while preserving shared candidates."""
    if not samples:
        raise ValueError("samples must not be empty")

    reference_shape = candidate_pool_shape(samples[0].candidate_trajectories)
    for index, sample in enumerate(samples):
        shape = candidate_pool_shape(sample.candidate_trajectories)
        if shape != reference_shape:
            raise ValueError(
                f"samples[{index}].candidate_trajectories shape {shape} "
                f"does not match shared shape {reference_shape}"
            )

    return {
        "scene_ids": [sample.scene_id for sample in samples],
        "sample_ids": [sample.sample_id for sample in samples],
        "timestamps": [sample.timestamp for sample in samples],
        "coordinate_frames": [sample.coordinate_frame for sample in samples],
        "scene_summary_history": [sample.scene_summary_history for sample in samples],
        "candidate_trajectories": [sample.candidate_trajectories for sample in samples],
        "logged_ego_future": [sample.logged_ego_future for sample in samples],
        "future_agents": [sample.future_agents for sample in samples],
        "labels": {
            "collision": [sample.labels.collision for sample in samples],
            "violation": [sample.labels.violation for sample in samples],
            "pseudo_sim_score": [sample.labels.pseudo_sim_score for sample in samples],
            "progress": [sample.labels.progress for sample in samples],
        },
        "candidate_pool_shape": reference_shape,
    }
