"""Map-grounded rule-violation risk head."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from coordiworld.risks.geometry import (
    distance_point_to_polyline,
    point_in_polygon,
    sigmoid_soft_margin,
    smooth_max,
    trajectory_crosses_polyline,
)
from coordiworld.scene_summary.schema import MapToken

Trajectory = Sequence[Sequence[float]]


@dataclass(frozen=True)
class RuleViolationConfig:
    lane_margin_m: float = 2.0
    stop_speed_threshold_mps: float = 0.2
    sigmoid_scale: float = 0.5
    smooth_max_temperature: float = 0.2


@dataclass(frozen=True)
class RuleViolationResult:
    p_violation: float
    components: dict[str, float]
    per_step_probabilities: list[float]


def compute_rule_violation_risk(
    ego_trajectory: Trajectory,
    map_tokens: Sequence[MapToken],
    *,
    config: RuleViolationConfig | None = None,
) -> RuleViolationResult:
    """Compute lane, drivable-area, stop-line, and traffic-light violation risk."""
    cfg = config or RuleViolationConfig()
    if not ego_trajectory:
        raise ValueError("ego_trajectory must not be empty")

    step_risks: list[float] = []
    lane_risks: list[float] = []
    drivable_risks: list[float] = []
    stop_line_risks: list[float] = []
    traffic_light_risks: list[float] = []

    for step_index, pose in enumerate(ego_trajectory):
        point = (float(pose[0]), float(pose[1]))
        lane_risk = _lane_departure_risk(point, map_tokens, cfg)
        drivable_risk = _drivable_area_risk(point, map_tokens, cfg)
        crossing_risk = _crossing_rule_risk(ego_trajectory, step_index, map_tokens, cfg)
        traffic_risk = _traffic_light_risk(ego_trajectory, step_index, map_tokens, cfg)

        lane_risks.append(lane_risk)
        drivable_risks.append(drivable_risk)
        stop_line_risks.append(crossing_risk)
        traffic_light_risks.append(traffic_risk)
        step_risks.append(
            smooth_max(
                [lane_risk, drivable_risk, crossing_risk, traffic_risk],
                temperature=cfg.smooth_max_temperature,
            )
        )

    components = {
        "lane_departure": smooth_max(lane_risks, temperature=cfg.smooth_max_temperature),
        "drivable_area": smooth_max(drivable_risks, temperature=cfg.smooth_max_temperature),
        "stop_line": smooth_max(stop_line_risks, temperature=cfg.smooth_max_temperature),
        "traffic_light": smooth_max(traffic_light_risks, temperature=cfg.smooth_max_temperature),
    }
    return RuleViolationResult(
        p_violation=smooth_max(step_risks, temperature=cfg.smooth_max_temperature),
        components=components,
        per_step_probabilities=step_risks,
    )


def _lane_departure_risk(
    point: tuple[float, float],
    map_tokens: Sequence[MapToken],
    config: RuleViolationConfig,
) -> float:
    lane_tokens = [
        token
        for token in map_tokens
        if token.type == "lane_centerline" and token.polyline
    ]
    if not lane_tokens:
        return 0.0
    nearest = min(distance_point_to_polyline(point, token.polyline or []) for token in lane_tokens)
    return sigmoid_soft_margin(nearest - config.lane_margin_m, scale=config.sigmoid_scale)


def _drivable_area_risk(
    point: tuple[float, float],
    map_tokens: Sequence[MapToken],
    config: RuleViolationConfig,
) -> float:
    drivable_tokens = [
        token
        for token in map_tokens
        if token.type == "drivable_area" and token.polygon
    ]
    if not drivable_tokens:
        return 0.0
    inside = any(point_in_polygon(point, token.polygon or []) for token in drivable_tokens)
    return 0.0 if inside else sigmoid_soft_margin(1.0, scale=config.sigmoid_scale)


def _crossing_rule_risk(
    ego_trajectory: Trajectory,
    step_index: int,
    map_tokens: Sequence[MapToken],
    config: RuleViolationConfig,
) -> float:
    if step_index == 0:
        return 0.0
    stop_lines = [token for token in map_tokens if token.type == "stop_line" and token.polyline]
    if not stop_lines:
        return 0.0
    previous_pose = ego_trajectory[step_index - 1]
    current_pose = ego_trajectory[step_index]
    speed = _step_speed(previous_pose, current_pose)
    crosses = any(
        trajectory_crosses_polyline(previous_pose, current_pose, token.polyline or [])
        for token in stop_lines
    )
    if not crosses:
        return 0.0
    return sigmoid_soft_margin(
        speed - config.stop_speed_threshold_mps,
        scale=config.sigmoid_scale,
    )


def _traffic_light_risk(
    ego_trajectory: Trajectory,
    step_index: int,
    map_tokens: Sequence[MapToken],
    config: RuleViolationConfig,
) -> float:
    if step_index == 0:
        return 0.0
    active_lights = [
        token
        for token in map_tokens
        if token.type == "traffic_light" and token.traffic_state in {"red", "yellow"}
    ]
    if not active_lights:
        return 0.0
    previous_pose = ego_trajectory[step_index - 1]
    current_pose = ego_trajectory[step_index]
    crossed = False
    for token in active_lights:
        if token.polyline and trajectory_crosses_polyline(
            previous_pose,
            current_pose,
            token.polyline,
        ):
            crossed = True
        if token.polygon and point_in_polygon(
            (float(current_pose[0]), float(current_pose[1])),
            token.polygon,
        ):
            crossed = True
    if not crossed:
        return 0.0
    return sigmoid_soft_margin(_step_speed(previous_pose, current_pose), scale=config.sigmoid_scale)


def _step_speed(previous_pose: Sequence[float], current_pose: Sequence[float]) -> float:
    return math.hypot(
        float(current_pose[0]) - float(previous_pose[0]),
        float(current_pose[1]) - float(previous_pose[1]),
    )
