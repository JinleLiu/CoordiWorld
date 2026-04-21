"""Tests for deterministic shared candidate pool construction."""

from __future__ import annotations

import math

import pytest

from coordiworld.data.base import candidate_pool_shape
from coordiworld.data.candidate_pool import (
    CandidatePoolConfig,
    build_candidate_pool,
    candidate_pool_config_from_mapping,
)


def test_candidate_pool_contains_required_variant_families() -> None:
    config = CandidatePoolConfig(
        speed_scaled=(0.5, 1.5),
        lateral_shift=(-0.5, 0.5),
        curvature_perturbed=(-0.01, 0.01),
        horizon_steps=4,
        nominal_speed_mps=4.0,
    )

    pool = build_candidate_pool(config)

    assert pool.shape == (7, 4, 3)
    assert candidate_pool_shape(pool.trajectories) == (7, 4, 3)
    assert [variant["name"] for variant in pool.metadata["variants"]] == [
        "nominal",
        "speed_scaled",
        "speed_scaled",
        "lateral_shift",
        "lateral_shift",
        "curvature_perturbed",
        "curvature_perturbed",
    ]


def test_nominal_and_speed_scaled_variants_are_deterministic() -> None:
    config = CandidatePoolConfig(
        speed_scaled=(2.0,),
        lateral_shift=(),
        curvature_perturbed=(),
        horizon_steps=3,
        step_time_s=1.0,
        nominal_speed_mps=2.0,
        seed=42,
    )

    first = build_candidate_pool(config)
    second = build_candidate_pool(config)

    assert first == second
    assert first.trajectories[0] == [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    assert first.trajectories[1][-1][0] == 12.0
    assert first.metadata["seed"] == 42


def test_lateral_shift_changes_y_without_changing_yaw() -> None:
    pool = build_candidate_pool(
        CandidatePoolConfig(
            nominal=False,
            speed_scaled=(),
            lateral_shift=(1.25,),
            curvature_perturbed=(),
            horizon_steps=2,
        )
    )

    assert [pose[1] for pose in pool.trajectories[0]] == [1.25, 1.25]
    assert [pose[2] for pose in pool.trajectories[0]] == [0.0, 0.0]


def test_curvature_perturbation_changes_lateral_position_and_yaw() -> None:
    pool = build_candidate_pool(
        CandidatePoolConfig(
            nominal=False,
            speed_scaled=(),
            lateral_shift=(),
            curvature_perturbed=(0.02,),
            horizon_steps=3,
        )
    )

    trajectory = pool.trajectories[0]
    assert trajectory[-1][1] > trajectory[0][1]
    assert trajectory[-1][2] > 0.0
    assert all(math.isfinite(value) for pose in trajectory for value in pose)


def test_candidate_pool_rejects_empty_variant_set() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_candidate_pool(
            CandidatePoolConfig(
                nominal=False,
                speed_scaled=(),
                lateral_shift=(),
                curvature_perturbed=(),
            )
        )


def test_candidate_pool_config_from_mapping_matches_protocol_fields() -> None:
    config = candidate_pool_config_from_mapping(
        {
            "nominal": True,
            "speed_scaled": [0.8, 1.2],
            "lateral_shift": [-1.0, 1.0],
            "curvature_perturbed": [-0.05, 0.05],
            "horizon_steps": 3,
        }
    )

    pool = build_candidate_pool(config)

    assert pool.shape == (7, 3, 3)
