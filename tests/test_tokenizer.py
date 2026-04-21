"""Tests for SceneSummary and candidate trajectory tokenizers."""

from __future__ import annotations

from dataclasses import replace

from coordiworld.scene_summary.schema import AgentState, EgoState, MapToken, SceneSummary
from coordiworld.tokens.action_tokenizer import ActionTokenizer
from coordiworld.tokens.scene_tokenizer import AGENT_FEATURE_DIM, EGO_FEATURE_DIM, SceneTokenizer


def make_agent(
    agent_id: str,
    *,
    x: float,
    y: float,
    confidence: float = 0.8,
    existence_prob: float = 0.9,
    flags: list[str] | None = None,
) -> AgentState:
    return AgentState(
        id=agent_id,
        type="vehicle",
        x=x,
        y=y,
        yaw=0.0,
        vx=1.0,
        vy=0.0,
        length=4.0,
        width=2.0,
        confidence=confidence,
        covariance_xy=[[0.4, 0.0], [0.0, 0.5]],
        existence_prob=existence_prob,
        source_modalities=["synthetic"],
        source_ids=[agent_id],
        fusion_lineage=["synthetic"],
        ambiguity_flags=flags or [],
        semantic_attributes={},
    )


def make_map_token(token_id: str, *, x: float, token_type: str = "lane_centerline") -> MapToken:
    return MapToken(
        id=token_id,
        type=token_type,
        polyline=[[x, 0.0], [x + 1.0, 0.0]],
        polygon=None,
        traffic_state="red" if token_type == "traffic_light" else None,
        rule_attributes={"fixture": True},
    )


def make_summary(
    *,
    agents: list[AgentState],
    map_tokens: list[MapToken] | None = None,
    timestamp: float = 0.0,
) -> SceneSummary:
    return SceneSummary(
        scene_id="tokenizer-scene",
        timestamp=timestamp,
        coordinate_frame="ego",
        ego=EgoState(x=0.0, y=0.0, yaw=0.0, vx=2.0, vy=0.0, length=4.8, width=2.0),
        agents=agents,
        map_tokens=map_tokens or [make_map_token("lane-0", x=0.0)],
        provenance={"dataset": "synthetic"},
        metadata={"fixture": "tokenizer"},
    )


def test_scene_tokenizer_shapes_and_masks() -> None:
    summary = make_summary(
        agents=[
            make_agent("a", x=5.0, y=0.0),
            make_agent("b", x=8.0, y=1.0),
        ]
    )

    tokenized = SceneTokenizer().tokenize(summary)

    assert len(tokenized.ego_tensor) == EGO_FEATURE_DIM
    assert len(tokenized.agent_tensor) == 32
    assert len(tokenized.agent_tensor[0]) == AGENT_FEATURE_DIM
    assert len(tokenized.map_tensor) == 24
    assert sum(tokenized.masks["agents"]) == 2
    assert sum(tokenized.masks["map_tokens"]) == 1
    assert tokenized.local_slot_indices == {"a": 0, "b": 1}


def test_agent_selection_keeps_near_and_risky_agents() -> None:
    agents = [
        make_agent(f"far-{index:02d}", x=100.0 + index, y=0.0, confidence=0.1)
        for index in range(40)
    ]
    agents.append(make_agent("near-agent", x=1.0, y=0.0, confidence=0.2))
    agents.append(
        make_agent(
            "risky-agent",
            x=20.0,
            y=0.0,
            confidence=1.0,
            existence_prob=1.0,
            flags=["class_conflict"],
        )
    )
    summary = make_summary(agents=agents)

    tokenized = SceneTokenizer().tokenize(summary)

    assert len(tokenized.local_slot_indices) == 32
    assert "near-agent" in tokenized.local_slot_indices
    assert "risky-agent" in tokenized.local_slot_indices
    assert "far-39" not in tokenized.local_slot_indices


def test_map_token_selection_uses_distance_and_rule_risk() -> None:
    map_tokens = [
        make_map_token(f"lane-{index:02d}", x=100.0 + index)
        for index in range(30)
    ]
    map_tokens.append(make_map_token("near-stop", x=5.0, token_type="stop_line"))
    map_tokens.append(make_map_token("traffic-risk", x=20.0, token_type="traffic_light"))
    summary = make_summary(agents=[], map_tokens=map_tokens)

    tokenized = SceneTokenizer().tokenize(summary)

    assert len(tokenized.map_tensor) == 24
    assert sum(tokenized.masks["map_tokens"]) == 24
    assert "near-stop" in tokenized.selected_map_ids
    assert "traffic-risk" in tokenized.selected_map_ids
    assert "lane-29" not in tokenized.selected_map_ids


def test_scene_local_slot_remapping_is_consistent_within_clip() -> None:
    first = make_summary(
        timestamp=0.0,
        agents=[
            make_agent("track-b", x=5.0, y=0.0),
            make_agent("track-a", x=3.0, y=0.0),
        ],
    )
    second = replace(
        first,
        timestamp=0.1,
        agents=[
            make_agent("track-a", x=4.0, y=0.0),
            make_agent("track-b", x=6.0, y=0.0),
        ],
    )

    tokenized = SceneTokenizer().tokenize([first, second])

    assert tokenized.local_slot_indices["track-a"] == 0
    assert tokenized.local_slot_indices["track-b"] == 1
    slot_a = tokenized.local_slot_indices["track-a"]
    assert tokenized.agent_tensor[slot_a][0] == 4.0


def test_action_tokenizer_outputs_action_tensor_shape_and_deltas() -> None:
    candidates = [
        [[1.0, 0.0, 0.0], [2.0, 0.5, 0.1]],
        [[1.5, 0.0, 0.0], [3.0, -0.5, -0.2]],
    ]

    tokenized = ActionTokenizer().tokenize(candidates)

    assert tokenized.shape == (2, 2, 6)
    assert tokenized.mask == [[1, 1], [1, 1]]
    assert tokenized.action_tensor[0][0] == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert tokenized.action_tensor[0][1] == [2.0, 0.5, 0.1, 1.0, 0.5, 0.1]
