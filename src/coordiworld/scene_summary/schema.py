"""Dataclass schema for structured SceneSummary records."""

from __future__ import annotations

from dataclasses import dataclass

VALID_MAP_TOKEN_TYPES: frozenset[str] = frozenset(
    {
        "lane_centerline",
        "drivable_area",
        "stop_line",
        "traffic_light",
        "conflict_zone",
    }
)


@dataclass
class EgoState:
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    length: float
    width: float


@dataclass
class AgentState:
    id: str
    type: str
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    length: float
    width: float
    confidence: float
    covariance_xy: list[list[float]]
    existence_prob: float
    source_modalities: list[str]
    source_ids: list[str]
    fusion_lineage: list[str]
    ambiguity_flags: list[str]
    semantic_attributes: dict[str, object]


@dataclass
class MapToken:
    id: str
    type: str
    polyline: list[list[float]] | None
    polygon: list[list[float]] | None
    traffic_state: str | None
    rule_attributes: dict[str, object]


@dataclass
class SceneSummary:
    scene_id: str
    timestamp: float
    coordinate_frame: str
    ego: EgoState
    agents: list[AgentState]
    map_tokens: list[MapToken]
    provenance: dict[str, object]
    metadata: dict[str, object]
