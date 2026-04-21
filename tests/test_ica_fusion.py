"""Synthetic ICA-style SceneSummary generation tests."""

from __future__ import annotations

import math

from coordiworld.scene_summary.generator import (
    CameraSemanticFact,
    GeometryFact,
    MultiSourceFacts,
    RadarFact,
    generate_scene_summary,
)
from coordiworld.scene_summary.schema import EgoState
from coordiworld.scene_summary.validators import validate_scene_summary


def make_ego() -> EgoState:
    return EgoState(x=0.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, length=4.8, width=2.0)


def make_geometry_fact(
    *,
    source_id: str,
    modality: str,
    type: str,
    x: float,
    y: float,
    vx: float = 0.0,
    vy: float = 0.0,
    confidence: float = 0.9,
) -> GeometryFact:
    return GeometryFact(
        source_id=source_id,
        modality=modality,
        type=type,
        x=x,
        y=y,
        yaw=0.0,
        vx=vx,
        vy=vy,
        length=4.5,
        width=1.9,
        confidence=confidence,
        covariance_xy=[[0.5, 0.0], [0.0, 0.5]],
        semantic_attributes={},
    )


def make_facts(
    *,
    lidar_objects: list[GeometryFact] | None = None,
    bevfusion_objects: list[GeometryFact] | None = None,
    radar_objects: list[RadarFact] | None = None,
    camera_objects: list[CameraSemanticFact] | None = None,
) -> MultiSourceFacts:
    return MultiSourceFacts(
        scene_id="synthetic-ica",
        timestamp=1.0,
        coordinate_frame="ego",
        ego=make_ego(),
        lidar_objects=lidar_objects or [],
        bevfusion_objects=bevfusion_objects or [],
        radar_objects=radar_objects or [],
        camera_objects=camera_objects or [],
        metadata={"fixture": "ica"},
    )


def camera_fact_for_agent(
    *,
    source_id: str = "camera-1",
    bbox_xyxy: list[float] | None = None,
    type: str | None = "sedan",
    color: str = "blue",
) -> CameraSemanticFact:
    return CameraSemanticFact(
        source_id=source_id,
        bbox_xyxy=bbox_xyxy or [9.5, -0.5, 10.5, 0.5],
        type=type,
        semantic_attributes={"color": color},
        confidence=0.8,
        camera_from_ego=[
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        intrinsic=[
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


def test_lidar_and_bevfusion_same_entity_conflict_resolved() -> None:
    facts = make_facts(
        lidar_objects=[
            make_geometry_fact(
                source_id="lidar-1",
                modality="lidar",
                type="vehicle",
                x=10.0,
                y=0.0,
                confidence=0.9,
            )
        ],
        bevfusion_objects=[
            make_geometry_fact(
                source_id="bev-1",
                modality="bevfusion",
                type="car",
                x=10.2,
                y=0.1,
                confidence=0.8,
            )
        ],
    )

    summary = generate_scene_summary(facts)

    validate_scene_summary(summary)
    assert len(summary.agents) == 1
    agent = summary.agents[0]
    assert set(agent.source_modalities) == {"bevfusion", "lidar"}
    assert set(agent.source_ids) == {"bev-1", "lidar-1"}
    assert "conflict_resolved" in agent.ambiguity_flags
    assert "class_conflict" in agent.ambiguity_flags


def test_radar_only_updates_velocity_not_geometry() -> None:
    facts = make_facts(
        lidar_objects=[
            make_geometry_fact(
                source_id="lidar-1",
                modality="lidar",
                type="vehicle",
                x=10.0,
                y=0.0,
            )
        ],
        radar_objects=[
            RadarFact(
                source_id="radar-1",
                range_m=10.0,
                azimuth_rad=0.0,
                radial_velocity_mps=5.0,
                confidence=1.0,
            )
        ],
    )

    summary = generate_scene_summary(facts)

    agent = summary.agents[0]
    assert agent.x == 10.0
    assert agent.y == 0.0
    assert agent.vx == 5.0
    assert agent.vy == 0.0
    assert "radar" in agent.source_modalities
    assert "radar_velocity_attachment" in agent.fusion_lineage


def test_camera_only_updates_semantics_not_geometry() -> None:
    facts = make_facts(
        lidar_objects=[
            make_geometry_fact(
                source_id="lidar-1",
                modality="lidar",
                type="vehicle",
                x=10.0,
                y=0.0,
            )
        ],
        camera_objects=[camera_fact_for_agent()],
    )

    summary = generate_scene_summary(facts)

    agent = summary.agents[0]
    assert agent.x == 10.0
    assert agent.y == 0.0
    assert agent.type == "vehicle"
    assert agent.semantic_attributes["color"] == "blue"
    assert agent.semantic_attributes["camera_type"] == "sedan"
    assert "camera" in agent.source_modalities
    assert "camera_semantic_attachment" in agent.fusion_lineage


def test_unmatched_camera_only_detection_is_not_exported_as_agent() -> None:
    facts = make_facts(
        camera_objects=[
            camera_fact_for_agent(
                source_id="camera-only",
                bbox_xyxy=[100.0, 100.0, 120.0, 120.0],
                type="pedestrian",
            )
        ]
    )

    summary = generate_scene_summary(facts)

    assert summary.agents == []
    assert summary.metadata["camera_only_export"] is False


def test_generator_is_deterministic_for_same_input() -> None:
    facts = make_facts(
        lidar_objects=[
            make_geometry_fact(
                source_id="lidar-1",
                modality="lidar",
                type="vehicle",
                x=10.0,
                y=0.0,
            )
        ],
        bevfusion_objects=[
            make_geometry_fact(
                source_id="bev-1",
                modality="bevfusion",
                type="car",
                x=10.2,
                y=0.1,
            )
        ],
        radar_objects=[RadarFact("radar-1", 10.0, 0.0, 3.0)],
        camera_objects=[camera_fact_for_agent()],
    )

    first = generate_scene_summary(facts)
    second = generate_scene_summary(facts)

    assert first == second


def test_radar_polar_projection_allows_offset_radar_attachment() -> None:
    facts = make_facts(
        lidar_objects=[
            make_geometry_fact(
                source_id="lidar-1",
                modality="lidar",
                type="vehicle",
                x=0.0,
                y=10.0,
            )
        ],
        radar_objects=[RadarFact("radar-1", 10.0, math.pi / 2.0, 2.0)],
    )

    summary = generate_scene_summary(facts)

    assert len(summary.agents) == 1
    assert abs(summary.agents[0].vy - 2.0) < 1e-9
