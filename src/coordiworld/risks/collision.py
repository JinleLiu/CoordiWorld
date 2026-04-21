"""Collision risk head for candidate-conditioned rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from coordiworld.risks.geometry import (
    box_from_pose,
    box_interaction_feature,
    clip01,
    noisy_or,
)

Trajectory = Sequence[Sequence[float]]


@dataclass(frozen=True)
class CollisionRiskConfig:
    ego_length: float = 4.8
    ego_width: float = 2.0
    default_agent_length: float = 4.5
    default_agent_width: float = 1.9
    collision_margin_m: float = 0.0
    sigmoid_scale_m: float = 0.75


@dataclass(frozen=True)
class CollisionRiskResult:
    p_collision: float
    per_step_probabilities: list[float]
    per_agent_step_probabilities: list[list[float]]


def compute_collision_risk(
    ego_trajectory: Trajectory,
    predicted_agent_trajectories: Sequence[Trajectory],
    *,
    existence_probabilities: Sequence[Sequence[float]] | None = None,
    agent_sizes: Sequence[tuple[float, float]] | None = None,
    config: CollisionRiskConfig | None = None,
) -> CollisionRiskResult:
    """Compute geometry-aware collision risk with noisy-OR plan aggregation."""
    cfg = config or CollisionRiskConfig()
    if not ego_trajectory:
        raise ValueError("ego_trajectory must not be empty")

    per_agent_step: list[list[float]] = []
    for agent_index, agent_trajectory in enumerate(predicted_agent_trajectories):
        if len(agent_trajectory) != len(ego_trajectory):
            raise ValueError("agent trajectories must share ego trajectory horizon")
        length, width = _agent_size(agent_index, agent_trajectory, agent_sizes, cfg)
        agent_step_probabilities: list[float] = []
        for step_index, (ego_pose, agent_pose) in enumerate(zip(ego_trajectory, agent_trajectory)):
            ego_box = box_from_pose(ego_pose, length=cfg.ego_length, width=cfg.ego_width)
            agent_box = box_from_pose(agent_pose, length=length, width=width)
            interaction = box_interaction_feature(
                ego_box,
                agent_box,
                margin=cfg.collision_margin_m,
                sigmoid_scale=cfg.sigmoid_scale_m,
            )
            existence = _existence(existence_probabilities, agent_index, step_index)
            agent_step_probabilities.append(
                clip01(interaction.soft_collision_probability * existence)
            )
        per_agent_step.append(agent_step_probabilities)

    per_step = [
        noisy_or([agent_probs[step_index] for agent_probs in per_agent_step])
        for step_index in range(len(ego_trajectory))
    ]
    return CollisionRiskResult(
        p_collision=noisy_or(per_step),
        per_step_probabilities=per_step,
        per_agent_step_probabilities=per_agent_step,
    )


def _agent_size(
    agent_index: int,
    agent_trajectory: Trajectory,
    agent_sizes: Sequence[tuple[float, float]] | None,
    config: CollisionRiskConfig,
) -> tuple[float, float]:
    if agent_sizes is not None:
        return agent_sizes[agent_index]
    first_pose = agent_trajectory[0]
    if len(first_pose) >= 5:
        return float(first_pose[3]), float(first_pose[4])
    return config.default_agent_length, config.default_agent_width


def _existence(
    existence_probabilities: Sequence[Sequence[float]] | None,
    agent_index: int,
    step_index: int,
) -> float:
    if existence_probabilities is None:
        return 1.0
    return clip01(float(existence_probabilities[agent_index][step_index]))
