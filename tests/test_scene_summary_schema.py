"""Synthetic tests for the SceneSummary schema, JSON I/O, and validator."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from coordiworld.scene_summary import (
    AgentState,
    EgoState,
    MapToken,
    SceneSummary,
    load_scene_summary_json,
    save_scene_summary_json,
    scene_summary_from_json,
    scene_summary_to_dict,
    scene_summary_to_json,
    validate_scene_summary,
)


def make_scene_summary() -> SceneSummary:
    return SceneSummary(
        scene_id="synthetic-scene-001",
        timestamp=12.5,
        coordinate_frame="ego_aligned_meters",
        ego=EgoState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            vx=4.0,
            vy=0.0,
            length=4.8,
            width=2.0,
        ),
        agents=[
            AgentState(
                id="agent-1",
                type="vehicle",
                x=12.0,
                y=1.5,
                yaw=0.1,
                vx=2.0,
                vy=0.0,
                length=4.5,
                width=1.9,
                confidence=0.92,
                covariance_xy=[[0.4, 0.0], [0.0, 0.6]],
                existence_prob=0.98,
                source_modalities=["lidar", "radar"],
                source_ids=["lidar-track-1", "radar-track-4"],
                fusion_lineage=["geometry_seed", "radar_velocity_attachment"],
                ambiguity_flags=[],
                semantic_attributes={"turn_signal": "unknown"},
            )
        ],
        map_tokens=[
            MapToken(
                id="lane-1",
                type="lane_centerline",
                polyline=[[0.0, 0.0], [10.0, 0.0], [20.0, 0.5]],
                polygon=None,
                traffic_state=None,
                rule_attributes={"speed_limit_mps": 13.4},
            ),
            MapToken(
                id="stop-line-1",
                type="stop_line",
                polyline=[[18.0, -2.0], [18.0, 2.0]],
                polygon=None,
                traffic_state="red",
                rule_attributes={"requires_stop": True},
            ),
        ],
        provenance={"sources": ["synthetic"]},
        metadata={"fixture": True},
    )


def test_valid_scene_summary_passes_validation() -> None:
    validate_scene_summary(make_scene_summary())


def test_json_roundtrip_preserves_fields() -> None:
    summary = make_scene_summary()

    loaded = scene_summary_from_json(scene_summary_to_json(summary, indent=2))

    validate_scene_summary(loaded)
    assert scene_summary_to_dict(loaded) == scene_summary_to_dict(summary)


def test_duplicate_agent_id_raises_value_error() -> None:
    summary = make_scene_summary()
    summary.agents.append(replace(summary.agents[0], x=20.0))

    with pytest.raises(ValueError, match="duplicate id"):
        validate_scene_summary(summary)


def test_duplicate_map_token_id_raises_value_error() -> None:
    summary = make_scene_summary()
    summary.map_tokens.append(replace(summary.map_tokens[0], type="drivable_area", polygon=[]))

    with pytest.raises(ValueError, match="duplicate id"):
        validate_scene_summary(summary)


def test_invalid_confidence_raises_value_error() -> None:
    summary = make_scene_summary()
    summary.agents[0].confidence = 1.1

    with pytest.raises(ValueError, match="confidence"):
        validate_scene_summary(summary)


def test_invalid_existence_prob_raises_value_error() -> None:
    summary = make_scene_summary()
    summary.agents[0].existence_prob = -0.1

    with pytest.raises(ValueError, match="existence_prob"):
        validate_scene_summary(summary)


def test_nan_coordinate_raises_value_error() -> None:
    summary = make_scene_summary()
    summary.agents[0].x = math.nan

    with pytest.raises(ValueError, match="finite number"):
        validate_scene_summary(summary)


@pytest.mark.parametrize("field_name", ["length", "width"])
def test_ego_non_positive_extent_raises_value_error(field_name: str) -> None:
    summary = make_scene_summary()
    setattr(summary.ego, field_name, 0.0)

    with pytest.raises(ValueError, match=f"ego.{field_name}"):
        validate_scene_summary(summary)


def test_agent_covariance_xy_not_2x2_raises_value_error() -> None:
    summary = make_scene_summary()
    summary.agents[0].covariance_xy = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    with pytest.raises(ValueError, match="2x2"):
        validate_scene_summary(summary)


def test_agent_covariance_xy_negative_diagonal_raises_value_error() -> None:
    summary = make_scene_summary()
    summary.agents[0].covariance_xy = [[-0.1, 0.0], [0.0, 1.0]]

    with pytest.raises(ValueError, match="non-negative"):
        validate_scene_summary(summary)


def test_invalid_map_token_type_raises_value_error() -> None:
    summary = make_scene_summary()
    summary.map_tokens[0].type = "crosswalk"

    with pytest.raises(ValueError, match="type"):
        validate_scene_summary(summary)


def test_map_token_without_geometry_raises_value_error() -> None:
    summary = make_scene_summary()
    summary.map_tokens[0].polyline = None
    summary.map_tokens[0].polygon = None

    with pytest.raises(ValueError, match="polyline or polygon"):
        validate_scene_summary(summary)


def test_map_token_geometry_point_not_xy_raises_value_error() -> None:
    summary = make_scene_summary()
    summary.map_tokens[0].polyline = [[0.0, 0.0, 1.0]]

    with pytest.raises(ValueError, match=r"\[x, y\]"):
        validate_scene_summary(summary)


def test_save_and_load_scene_summary_json(tmp_path: Path) -> None:
    summary = make_scene_summary()
    path = tmp_path / "scene_summary.json"

    save_scene_summary_json(summary, path)
    loaded = load_scene_summary_json(path)

    validate_scene_summary(loaded)
    assert scene_summary_to_dict(loaded) == scene_summary_to_dict(summary)
