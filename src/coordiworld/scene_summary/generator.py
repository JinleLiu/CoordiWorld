"""ICA-style deterministic SceneSummary generator for synthetic multi-source facts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

from coordiworld.scene_summary.association import associate_by_bev
from coordiworld.scene_summary.fusion import (
    build_fusion_trace,
    categorical_weighted_vote,
    collect_semantic_conflicts,
    continuous_weighted_fusion,
    generate_ambiguity_flags,
    merge_semantic_attributes,
)
from coordiworld.scene_summary.schema import AgentState, EgoState, MapToken, SceneSummary
from coordiworld.scene_summary.transforms import (
    Matrix,
    point_in_bbox,
    project_ego_point_to_camera,
    radar_polar_to_ego_bev,
    transform_point_sensor_to_ego,
    transform_velocity_sensor_to_ego,
    transform_yaw_sensor_to_ego,
)
from coordiworld.scene_summary.validators import validate_scene_summary


@dataclass(frozen=True)
class GeometryFact:
    source_id: str
    modality: str
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
    existence_prob: float = 1.0
    semantic_attributes: dict[str, object] = field(default_factory=dict)
    sensor_to_ego: Matrix | None = None


@dataclass(frozen=True)
class RadarFact:
    source_id: str
    range_m: float
    azimuth_rad: float
    radial_velocity_mps: float
    confidence: float = 1.0
    modality: str = "radar"
    sensor_to_ego: Matrix | None = None


@dataclass(frozen=True)
class CameraSemanticFact:
    source_id: str
    bbox_xyxy: list[float]
    type: str | None
    semantic_attributes: dict[str, object]
    confidence: float
    camera_from_ego: Matrix
    intrinsic: Matrix
    modality: str = "camera"


@dataclass(frozen=True)
class MultiSourceFacts:
    scene_id: str
    timestamp: float
    coordinate_frame: str
    ego: EgoState
    lidar_objects: list[GeometryFact] = field(default_factory=list)
    bevfusion_objects: list[GeometryFact] = field(default_factory=list)
    radar_objects: list[RadarFact] = field(default_factory=list)
    camera_objects: list[CameraSemanticFact] = field(default_factory=list)
    map_tokens: list[MapToken] = field(default_factory=list)
    provenance: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class _AgentCluster:
    geometry_facts: list[GeometryFact]
    radar_facts: list[RadarFact] = field(default_factory=list)
    camera_facts: list[CameraSemanticFact] = field(default_factory=list)


def generate_scene_summary(
    facts: MultiSourceFacts,
    *,
    geometry_distance_gate_m: float = 2.0,
    radar_distance_gate_m: float = 2.5,
) -> SceneSummary:
    """Generate a deterministic SceneSummary from normalized synthetic source facts."""
    normalized_lidar = [_normalize_geometry_fact(fact) for fact in facts.lidar_objects]
    normalized_bev = [_normalize_geometry_fact(fact) for fact in facts.bevfusion_objects]
    clusters = _build_geometry_seed_clusters(
        normalized_lidar,
        normalized_bev,
        distance_gate_m=geometry_distance_gate_m,
    )

    _attach_radar_velocity(clusters, facts.radar_objects, distance_gate_m=radar_distance_gate_m)
    _attach_camera_semantics(clusters, facts.camera_objects)

    agents = [
        _cluster_to_agent(cluster, agent_index=index)
        for index, cluster in enumerate(_sort_clusters(clusters))
    ]
    summary = SceneSummary(
        scene_id=facts.scene_id,
        timestamp=facts.timestamp,
        coordinate_frame=facts.coordinate_frame,
        ego=facts.ego,
        agents=agents,
        map_tokens=list(facts.map_tokens),
        provenance={
            **facts.provenance,
            "generator": "ica_style_scene_summary",
            "stages": [
                "coordinate_normalization",
                "geometry_seed_matching",
                "radar_velocity_attachment",
                "camera_semantic_attachment",
                "continuous_fusion",
                "categorical_voting",
                "ambiguity_flag_generation",
            ],
        },
        metadata={**facts.metadata, "camera_only_export": False},
    )
    validate_scene_summary(summary)
    return summary


def _normalize_geometry_fact(fact: GeometryFact) -> GeometryFact:
    x, y = transform_point_sensor_to_ego(fact.x, fact.y, fact.sensor_to_ego)
    vx, vy = transform_velocity_sensor_to_ego(fact.vx, fact.vy, fact.sensor_to_ego)
    yaw = transform_yaw_sensor_to_ego(fact.yaw, fact.sensor_to_ego)
    return replace(fact, x=x, y=y, vx=vx, vy=vy, yaw=yaw, sensor_to_ego=None)


def _build_geometry_seed_clusters(
    lidar_facts: Sequence[GeometryFact],
    bev_facts: Sequence[GeometryFact],
    *,
    distance_gate_m: float,
) -> list[_AgentCluster]:
    association = associate_by_bev(
        lidar_facts,
        bev_facts,
        distance_gate_m=distance_gate_m,
        mahalanobis_gate=3.0,
    )

    clusters: list[_AgentCluster] = []
    for match in association.matches:
        clusters.append(
            _AgentCluster(
                geometry_facts=[
                    lidar_facts[match.left_index],
                    bev_facts[match.right_index],
                ]
            )
        )
    for index in association.unmatched_left:
        clusters.append(_AgentCluster(geometry_facts=[lidar_facts[index]]))
    for index in association.unmatched_right:
        clusters.append(_AgentCluster(geometry_facts=[bev_facts[index]]))
    return clusters


def _attach_radar_velocity(
    clusters: list[_AgentCluster],
    radar_facts: Sequence[RadarFact],
    *,
    distance_gate_m: float,
) -> None:
    if not clusters or not radar_facts:
        return

    cluster_centers = [_cluster_center(cluster) for cluster in clusters]
    radar_points = [
        _radar_fact_to_point(fact)
        for fact in radar_facts
    ]
    association = associate_by_bev(
        cluster_centers,
        radar_points,
        distance_gate_m=distance_gate_m,
        mahalanobis_gate=3.0,
    )
    for match in association.matches:
        clusters[match.left_index].radar_facts.append(radar_facts[match.right_index])


def _attach_camera_semantics(
    clusters: list[_AgentCluster],
    camera_facts: Sequence[CameraSemanticFact],
) -> None:
    for camera_fact in camera_facts:
        matched_cluster: _AgentCluster | None = None
        matched_depth: float | None = None
        for cluster in _sort_clusters(clusters):
            center = _cluster_center(cluster)
            projection = project_ego_point_to_camera(
                center["x"],
                center["y"],
                0.0,
                camera_fact.camera_from_ego,
                camera_fact.intrinsic,
            )
            if projection is None:
                continue
            if not point_in_bbox(projection.u, projection.v, camera_fact.bbox_xyxy):
                continue
            if matched_depth is None or projection.depth < matched_depth:
                matched_cluster = cluster
                matched_depth = projection.depth
        if matched_cluster is not None:
            matched_cluster.camera_facts.append(camera_fact)


def _cluster_to_agent(cluster: _AgentCluster, *, agent_index: int) -> AgentState:
    geometry_facts = sorted(
        cluster.geometry_facts,
        key=lambda fact: (fact.modality, fact.source_id),
    )
    weights = [fact.confidence for fact in geometry_facts]

    x = continuous_weighted_fusion([fact.x for fact in geometry_facts], weights)
    y = continuous_weighted_fusion([fact.y for fact in geometry_facts], weights)
    yaw = continuous_weighted_fusion([fact.yaw for fact in geometry_facts], weights)
    vx = continuous_weighted_fusion([fact.vx for fact in geometry_facts], weights)
    vy = continuous_weighted_fusion([fact.vy for fact in geometry_facts], weights)

    if cluster.radar_facts:
        radar_points = [_radar_fact_to_point(fact) for fact in cluster.radar_facts]
        radar_weights = [fact.confidence for fact in cluster.radar_facts]
        vx = continuous_weighted_fusion([point["vx"] for point in radar_points], radar_weights)
        vy = continuous_weighted_fusion([point["vy"] for point in radar_points], radar_weights)

    all_facts = [*geometry_facts, *cluster.radar_facts, *cluster.camera_facts]
    semantic_attributes = merge_semantic_attributes(
        [
            fact.semantic_attributes
            for fact in [*geometry_facts, *cluster.camera_facts]
            if fact.semantic_attributes
        ]
    )
    for camera_fact in sorted(cluster.camera_facts, key=lambda fact: fact.source_id):
        if camera_fact.type:
            semantic_attributes.setdefault("camera_type", camera_fact.type)

    type_votes = [fact.type for fact in geometry_facts]
    selected_type = categorical_weighted_vote(type_votes, weights)
    semantic_conflicts = collect_semantic_conflicts([*geometry_facts, *cluster.camera_facts])
    ambiguity_flags = generate_ambiguity_flags(
        type_votes=type_votes,
        semantic_conflicts=semantic_conflicts,
    )
    stages = [
        "geometry_seed_matching",
        "continuous_fusion",
        "categorical_voting",
        "ambiguity_flag_generation",
    ]
    if cluster.radar_facts:
        stages.append("radar_velocity_attachment")
    if cluster.camera_facts:
        stages.append("camera_semantic_attachment")
    trace = build_fusion_trace(all_facts, stages=stages, ambiguity_flags=ambiguity_flags)

    length = continuous_weighted_fusion([fact.length for fact in geometry_facts], weights)
    width = continuous_weighted_fusion([fact.width for fact in geometry_facts], weights)
    confidence = max(0.0, min(1.0, continuous_weighted_fusion(weights, weights)))
    existence_prob = max(
        0.0,
        min(
            1.0,
            continuous_weighted_fusion(
                [fact.existence_prob for fact in geometry_facts],
                weights,
            ),
        ),
    )
    covariance_xy = _fuse_covariance(geometry_facts, weights)

    return AgentState(
        id=f"agent-{agent_index:03d}",
        type=selected_type,
        x=x,
        y=y,
        yaw=yaw,
        vx=vx,
        vy=vy,
        length=length,
        width=width,
        confidence=confidence,
        covariance_xy=covariance_xy,
        existence_prob=existence_prob,
        source_modalities=trace.source_modalities,
        source_ids=trace.source_ids,
        fusion_lineage=trace.fusion_lineage,
        ambiguity_flags=trace.ambiguity_flags,
        semantic_attributes=semantic_attributes,
    )


def _cluster_center(cluster: _AgentCluster) -> dict[str, float | str]:
    geometry_facts = cluster.geometry_facts
    weights = [fact.confidence for fact in geometry_facts]
    return {
        "x": continuous_weighted_fusion([fact.x for fact in geometry_facts], weights),
        "y": continuous_weighted_fusion([fact.y for fact in geometry_facts], weights),
        "type": categorical_weighted_vote([fact.type for fact in geometry_facts], weights),
    }


def _radar_fact_to_point(fact: RadarFact) -> dict[str, float | str]:
    point = radar_polar_to_ego_bev(
        fact.range_m,
        fact.azimuth_rad,
        fact.radial_velocity_mps,
        fact.sensor_to_ego,
    )
    return {"x": point.x, "y": point.y, "vx": point.vx, "vy": point.vy, "type": "unknown"}


def _sort_clusters(clusters: Sequence[_AgentCluster]) -> list[_AgentCluster]:
    return sorted(
        clusters,
        key=lambda cluster: (
            round(float(_cluster_center(cluster)["x"]), 6),
            round(float(_cluster_center(cluster)["y"]), 6),
            ",".join(sorted(fact.source_id for fact in cluster.geometry_facts)),
        ),
    )


def _fuse_covariance(
    geometry_facts: Sequence[GeometryFact],
    weights: Sequence[float],
) -> list[list[float]]:
    return [
        [
            continuous_weighted_fusion(
                [fact.covariance_xy[row][column] for fact in geometry_facts],
                weights,
            )
            for column in range(2)
        ]
        for row in range(2)
    ]
