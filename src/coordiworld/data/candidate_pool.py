"""Shared candidate trajectory pool construction for CoordiWorld."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from coordiworld.data.base import CandidateTrajectories, candidate_pool_shape


@dataclass(frozen=True)
class CandidatePoolConfig:
    nominal: bool = True
    speed_scaled: tuple[float, ...] = (0.75, 1.25)
    lateral_shift: tuple[float, ...] = (-1.0, 1.0)
    curvature_perturbed: tuple[float, ...] = (-0.02, 0.02)
    horizon_steps: int = 6
    step_time_s: float = 0.5
    nominal_speed_mps: float = 5.0
    seed: int = 0


@dataclass(frozen=True)
class CandidatePool:
    trajectories: CandidateTrajectories
    metadata: dict[str, object]

    @property
    def shape(self) -> tuple[int, int, int]:
        return candidate_pool_shape(self.trajectories)


def build_candidate_pool(config: CandidatePoolConfig | None = None) -> CandidatePool:
    """Build deterministic shared candidate trajectories with [M,H,3] shape."""
    cfg = config or CandidatePoolConfig()
    _validate_config(cfg)

    trajectories: CandidateTrajectories = []
    variants: list[dict[str, object]] = []

    if cfg.nominal:
        _append_variant(
            trajectories,
            variants,
            name="nominal",
            trajectory=_build_trajectory(cfg, speed_mps=cfg.nominal_speed_mps),
        )

    for scale in cfg.speed_scaled:
        _append_variant(
            trajectories,
            variants,
            name="speed_scaled",
            parameters={"scale": scale},
            trajectory=_build_trajectory(cfg, speed_mps=cfg.nominal_speed_mps * scale),
        )

    for offset_m in cfg.lateral_shift:
        _append_variant(
            trajectories,
            variants,
            name="lateral_shift",
            parameters={"offset_m": offset_m},
            trajectory=_build_trajectory(
                cfg,
                speed_mps=cfg.nominal_speed_mps,
                lateral_offset_m=offset_m,
            ),
        )

    for curvature in cfg.curvature_perturbed:
        _append_variant(
            trajectories,
            variants,
            name="curvature_perturbed",
            parameters={"curvature": curvature},
            trajectory=_build_trajectory(
                cfg,
                speed_mps=cfg.nominal_speed_mps,
                curvature=curvature,
            ),
        )

    if not trajectories:
        raise ValueError("CandidatePoolConfig must enable at least one candidate variant")

    return CandidatePool(
        trajectories=trajectories,
        metadata={
            "candidate_pool_type": "shared",
            "seed": cfg.seed,
            "shape": candidate_pool_shape(trajectories),
            "variants": variants,
            "units": {"x": "m", "y": "m", "yaw": "rad"},
        },
    )


def build_shared_candidate_pool(config: CandidatePoolConfig | None = None) -> CandidatePool:
    """Alias for evaluator protocol wording: all methods share this candidate set."""
    return build_candidate_pool(config)


def candidate_pool_config_from_mapping(data: dict[str, Any] | None) -> CandidatePoolConfig:
    """Build CandidatePoolConfig from config dictionaries used by CLIs/examples."""
    if not data:
        return CandidatePoolConfig()
    return CandidatePoolConfig(
        nominal=_as_bool(data.get("nominal", CandidatePoolConfig.nominal)),
        speed_scaled=_as_float_tuple(data.get("speed_scaled", CandidatePoolConfig.speed_scaled)),
        lateral_shift=_as_float_tuple(data.get("lateral_shift", CandidatePoolConfig.lateral_shift)),
        curvature_perturbed=_as_float_tuple(
            data.get("curvature_perturbed", CandidatePoolConfig.curvature_perturbed)
        ),
        horizon_steps=int(data.get("horizon_steps", CandidatePoolConfig.horizon_steps)),
        step_time_s=float(data.get("step_time_s", CandidatePoolConfig.step_time_s)),
        nominal_speed_mps=float(
            data.get("nominal_speed_mps", CandidatePoolConfig.nominal_speed_mps)
        ),
        seed=int(data.get("seed", CandidatePoolConfig.seed)),
    )


def _build_trajectory(
    config: CandidatePoolConfig,
    *,
    speed_mps: float,
    lateral_offset_m: float = 0.0,
    curvature: float = 0.0,
) -> list[list[float]]:
    trajectory: list[list[float]] = []
    for step in range(1, config.horizon_steps + 1):
        arc_length = float(speed_mps) * config.step_time_s * step
        x = arc_length
        y = float(lateral_offset_m) + 0.5 * float(curvature) * arc_length * arc_length
        yaw = math.atan(float(curvature) * arc_length)
        trajectory.append([x, y, yaw])
    return trajectory


def _append_variant(
    trajectories: CandidateTrajectories,
    variants: list[dict[str, object]],
    *,
    name: str,
    trajectory: list[list[float]],
    parameters: dict[str, object] | None = None,
) -> None:
    trajectories.append(trajectory)
    variants.append(
        {
            "index": len(trajectories) - 1,
            "name": name,
            "parameters": parameters or {},
        }
    )


def _validate_config(config: CandidatePoolConfig) -> None:
    if config.horizon_steps <= 0:
        raise ValueError("horizon_steps must be > 0")
    if config.step_time_s <= 0:
        raise ValueError("step_time_s must be > 0")
    if config.nominal_speed_mps < 0:
        raise ValueError("nominal_speed_mps must be >= 0")
    for scale in config.speed_scaled:
        if scale <= 0:
            raise ValueError("speed_scaled entries must be > 0")


def _as_float_tuple(value: object) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple(float(item) for item in value)
    if isinstance(value, list):
        return tuple(float(item) for item in value)
    return (float(value),)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
