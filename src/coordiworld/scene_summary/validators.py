"""Validation helpers for SceneSummary dataclasses."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from coordiworld.scene_summary.schema import (
    VALID_MAP_TOKEN_TYPES,
    AgentState,
    EgoState,
    MapToken,
    SceneSummary,
)

EGO_NUMERIC_FIELDS: tuple[str, ...] = ("x", "y", "yaw", "vx", "vy", "length", "width")
AGENT_NUMERIC_FIELDS: tuple[str, ...] = (
    "x",
    "y",
    "yaw",
    "vx",
    "vy",
    "length",
    "width",
)
AGENT_LIST_FIELDS: tuple[str, ...] = (
    "source_modalities",
    "source_ids",
    "fusion_lineage",
    "ambiguity_flags",
)


def validate_scene_summary(summary: SceneSummary) -> None:
    """Validate a SceneSummary and raise ValueError with a clear message on failure."""
    if not isinstance(summary, SceneSummary):
        raise ValueError("summary must be a SceneSummary")
    if not _is_nonempty_string(summary.scene_id):
        raise ValueError("scene_id must be a non-empty string")
    if not _is_nonempty_string(summary.coordinate_frame):
        raise ValueError("coordinate_frame must be a non-empty string")
    _require_finite_number(summary.timestamp, "timestamp")

    _validate_ego(summary.ego)
    _require_list(summary.agents, "agents")
    _require_list(summary.map_tokens, "map_tokens")
    _require_dict(summary.provenance, "provenance")
    _require_dict(summary.metadata, "metadata")

    seen_agent_ids: set[str] = set()
    for index, agent in enumerate(summary.agents):
        _validate_agent(agent, f"agents[{index}]")
        if agent.id in seen_agent_ids:
            raise ValueError(f"agents contains duplicate id: {agent.id!r}")
        seen_agent_ids.add(agent.id)

    seen_map_token_ids: set[str] = set()
    for index, token in enumerate(summary.map_tokens):
        _validate_map_token(token, f"map_tokens[{index}]")
        if token.id in seen_map_token_ids:
            raise ValueError(f"map_tokens contains duplicate id: {token.id!r}")
        seen_map_token_ids.add(token.id)


def _validate_ego(ego: EgoState) -> None:
    if not isinstance(ego, EgoState):
        raise ValueError("ego must be an EgoState")
    for field_name in EGO_NUMERIC_FIELDS:
        _require_finite_number(getattr(ego, field_name), f"ego.{field_name}")
    if ego.length <= 0:
        raise ValueError("ego.length must be > 0")
    if ego.width <= 0:
        raise ValueError("ego.width must be > 0")


def _validate_agent(agent: AgentState, path: str) -> None:
    if not isinstance(agent, AgentState):
        raise ValueError(f"{path} must be an AgentState")
    if not _is_nonempty_string(agent.id):
        raise ValueError(f"{path}.id must be a non-empty string")

    for field_name in AGENT_NUMERIC_FIELDS:
        _require_finite_number(getattr(agent, field_name), f"{path}.{field_name}")
    if agent.length <= 0:
        raise ValueError(f"{path}.length must be > 0")
    if agent.width <= 0:
        raise ValueError(f"{path}.width must be > 0")

    _require_unit_interval(agent.confidence, f"{path}.confidence")
    _require_unit_interval(agent.existence_prob, f"{path}.existence_prob")
    _validate_covariance_xy(agent.covariance_xy, f"{path}.covariance_xy")

    for field_name in AGENT_LIST_FIELDS:
        _require_list(getattr(agent, field_name), f"{path}.{field_name}")
    _require_dict(agent.semantic_attributes, f"{path}.semantic_attributes")


def _validate_map_token(token: MapToken, path: str) -> None:
    if not isinstance(token, MapToken):
        raise ValueError(f"{path} must be a MapToken")
    if not _is_nonempty_string(token.id):
        raise ValueError(f"{path}.id must be a non-empty string")
    if token.type not in VALID_MAP_TOKEN_TYPES:
        allowed = ", ".join(sorted(VALID_MAP_TOKEN_TYPES))
        raise ValueError(f"{path}.type must be one of: {allowed}")
    _require_dict(token.rule_attributes, f"{path}.rule_attributes")

    has_polyline = bool(token.polyline)
    has_polygon = bool(token.polygon)
    if not has_polyline and not has_polygon:
        raise ValueError(f"{path} must include a non-empty polyline or polygon")
    if token.polyline is not None:
        _validate_geometry(token.polyline, f"{path}.polyline")
    if token.polygon is not None:
        _validate_geometry(token.polygon, f"{path}.polygon")


def _validate_geometry(points: Any, path: str) -> None:
    if not isinstance(points, list):
        raise ValueError(f"{path} must be a list of [x, y] points")
    for index, point in enumerate(points):
        point_path = f"{path}[{index}]"
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{point_path} must be a [x, y] list")
        _require_finite_number(point[0], f"{point_path}[0]")
        _require_finite_number(point[1], f"{point_path}[1]")


def _validate_covariance_xy(value: Any, path: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{path} must be a 2x2 numeric matrix")
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{path} must be a 2x2 numeric matrix")
        for column_index, cell in enumerate(row):
            _require_finite_number(cell, f"{path}[{row_index}][{column_index}]")
    if value[0][0] < 0:
        raise ValueError(f"{path}[0][0] must be non-negative")
    if value[1][1] < 0:
        raise ValueError(f"{path}[1][1] must be non-negative")


def _require_unit_interval(value: Any, path: str) -> None:
    _require_finite_number(value, path)
    if value < 0 or value > 1:
        raise ValueError(f"{path} must be in [0, 1]")


def _require_finite_number(value: Any, path: str) -> None:
    if not _is_finite_number(value):
        raise ValueError(f"{path} must be a finite number")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_list(value: Any, path: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")


def _require_dict(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a dict")
