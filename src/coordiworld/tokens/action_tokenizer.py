"""Candidate trajectory tokenizer for fixed shared candidate sets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Sequence

ACTION_FEATURE_DIM = 6


@dataclass(frozen=True)
class TokenizedActions:
    action_tensor: list[list[list[float]]]
    mask: list[list[int]]
    shape: tuple[int, int, int]
    metadata: dict[str, object]


class ActionTokenizer:
    """Convert candidate trajectories [M,H,3] into action token features."""

    def tokenize(
        self,
        candidate_trajectories: Sequence[Sequence[Sequence[float]]],
    ) -> TokenizedActions:
        m_count, horizon = _validate_candidate_trajectories(candidate_trajectories)
        action_tensor: list[list[list[float]]] = []
        mask: list[list[int]] = []

        for trajectory in candidate_trajectories:
            candidate_tokens: list[list[float]] = []
            candidate_mask: list[int] = []
            prev_x = 0.0
            prev_y = 0.0
            prev_yaw = 0.0
            for pose in trajectory:
                x = float(pose[0])
                y = float(pose[1])
                yaw = float(pose[2])
                candidate_tokens.append(
                    [
                        x,
                        y,
                        yaw,
                        x - prev_x,
                        y - prev_y,
                        round(_wrap_angle(yaw - prev_yaw), 12),
                    ]
                )
                candidate_mask.append(1)
                prev_x = x
                prev_y = y
                prev_yaw = yaw
            action_tensor.append(candidate_tokens)
            mask.append(candidate_mask)

        return TokenizedActions(
            action_tensor=action_tensor,
            mask=mask,
            shape=(m_count, horizon, ACTION_FEATURE_DIM),
            metadata={"input_shape": (m_count, horizon, 3), "feature_order": feature_order()},
        )


def feature_order() -> list[str]:
    return ["x", "y", "yaw", "delta_x", "delta_y", "delta_yaw"]


def _validate_candidate_trajectories(
    candidate_trajectories: Sequence[Sequence[Sequence[float]]],
) -> tuple[int, int]:
    if not candidate_trajectories:
        raise ValueError("candidate_trajectories must be a non-empty [M,H,3] sequence")
    horizon = len(candidate_trajectories[0])
    if horizon == 0:
        raise ValueError("candidate_trajectories horizon must be > 0")
    for candidate_index, trajectory in enumerate(candidate_trajectories):
        if len(trajectory) != horizon:
            raise ValueError("candidate_trajectories must have a shared horizon")
        for step_index, pose in enumerate(trajectory):
            if len(pose) != 3:
                raise ValueError(
                    f"candidate_trajectories[{candidate_index}][{step_index}] must be [x, y, yaw]"
                )
            for value in pose:
                if not isinstance(value, Real) or isinstance(value, bool):
                    raise ValueError("candidate trajectory values must be numeric")
                if not math.isfinite(float(value)):
                    raise ValueError("candidate trajectory values must be finite")
    return len(candidate_trajectories), horizon


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
