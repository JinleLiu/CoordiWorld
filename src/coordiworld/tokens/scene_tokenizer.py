"""SceneSummary tokenizer for ego, agent, and map token groups."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Sequence

from coordiworld.scene_summary.schema import AgentState, SceneSummary
from coordiworld.tokens.map_tokenizer import (
    MAP_FEATURE_DIM,
    MapTokenizer,
    MapTokenizerConfig,
    map_feature_order,
)

AGENT_FEATURE_DIM = 16
EGO_FEATURE_DIM = 7
AGENT_TYPE_CODES: dict[str, float] = {
    "unknown": 0.0,
    "vehicle": 1.0,
    "car": 1.0,
    "truck": 1.1,
    "bus": 1.2,
    "pedestrian": 2.0,
    "person": 2.0,
    "cyclist": 3.0,
    "bicycle": 3.0,
}


@dataclass(frozen=True)
class SceneTokenizerConfig:
    max_agents: int = 32
    max_map_tokens: int = 24


@dataclass(frozen=True)
class TokenizedScene:
    ego_tensor: list[float]
    agent_tensor: list[list[float]]
    map_tensor: list[list[float]]
    masks: dict[str, list[int]]
    local_slot_indices: dict[str, int]
    selected_agent_ids: list[str]
    selected_map_ids: list[str]
    feature_metadata: dict[str, object]


class SceneTokenizer:
    """Tokenize a SceneSummary frame or clip into padded numeric groups."""

    def __init__(self, config: SceneTokenizerConfig | None = None) -> None:
        self.config = config or SceneTokenizerConfig()
        if self.config.max_agents <= 0:
            raise ValueError("max_agents must be > 0")
        if self.config.max_map_tokens <= 0:
            raise ValueError("max_map_tokens must be > 0")
        self._map_tokenizer = MapTokenizer(MapTokenizerConfig(self.config.max_map_tokens))

    def tokenize(self, scene_or_history: SceneSummary | Sequence[SceneSummary]) -> TokenizedScene:
        history = _as_history(scene_or_history)
        latest = history[-1]

        selected_agents = _select_agent_ids(history, self.config.max_agents)
        local_slot_indices = {agent_id: slot for slot, agent_id in enumerate(selected_agents)}
        latest_agents = {agent.id: agent for agent in latest.agents}

        agent_tensor = [[0.0] * AGENT_FEATURE_DIM for _ in range(self.config.max_agents)]
        agent_mask = [0 for _ in range(self.config.max_agents)]
        for agent_id, slot in local_slot_indices.items():
            agent = latest_agents.get(agent_id)
            if agent is None:
                continue
            agent_tensor[slot] = _encode_agent(agent, latest)
            agent_mask[slot] = 1

        tokenized_map = self._map_tokenizer.tokenize(latest.map_tokens, latest.ego)
        return TokenizedScene(
            ego_tensor=_encode_ego(latest),
            agent_tensor=agent_tensor,
            map_tensor=tokenized_map.map_tensor,
            masks={"agents": agent_mask, "map_tokens": tokenized_map.mask},
            local_slot_indices=local_slot_indices,
            selected_agent_ids=selected_agents,
            selected_map_ids=tokenized_map.selected_ids,
            feature_metadata={
                "ego_feature_order": ego_feature_order(),
                "agent_feature_order": agent_feature_order(),
                "map_feature_order": map_feature_order(),
                "map_feature_dim": MAP_FEATURE_DIM,
            },
        )


def ego_feature_order() -> list[str]:
    return ["x", "y", "yaw", "vx", "vy", "length", "width"]


def agent_feature_order() -> list[str]:
    return [
        "rel_x",
        "rel_y",
        "yaw",
        "vx",
        "vy",
        "length",
        "width",
        "confidence",
        "existence_prob",
        "cov_xx",
        "cov_xy",
        "cov_yy",
        "type_code",
        "source_count",
        "ambiguity_count",
        "risk_score",
    ]


def _as_history(scene_or_history: SceneSummary | Sequence[SceneSummary]) -> list[SceneSummary]:
    if isinstance(scene_or_history, SceneSummary):
        return [scene_or_history]
    history = list(scene_or_history)
    if not history:
        raise ValueError("scene summary history must not be empty")
    if not all(isinstance(summary, SceneSummary) for summary in history):
        raise ValueError("scene summary history must contain SceneSummary objects")
    return history


def _encode_ego(summary: SceneSummary) -> list[float]:
    ego = summary.ego
    return [ego.x, ego.y, ego.yaw, ego.vx, ego.vy, ego.length, ego.width]


def _encode_agent(agent: AgentState, summary: SceneSummary) -> list[float]:
    rel_x = agent.x - summary.ego.x
    rel_y = agent.y - summary.ego.y
    risk_score = _agent_risk_score(agent, summary)
    return [
        rel_x,
        rel_y,
        agent.yaw,
        agent.vx,
        agent.vy,
        agent.length,
        agent.width,
        agent.confidence,
        agent.existence_prob,
        _covariance_value(agent, 0, 0),
        _covariance_value(agent, 0, 1),
        _covariance_value(agent, 1, 1),
        AGENT_TYPE_CODES.get(agent.type.lower(), AGENT_TYPE_CODES["unknown"]),
        float(len(agent.source_modalities)),
        float(len(agent.ambiguity_flags)),
        risk_score,
    ]


def _select_agent_ids(history: Sequence[SceneSummary], max_agents: int) -> list[str]:
    best_agents: dict[str, tuple[tuple[float, float, str], AgentState]] = {}
    for summary in history:
        for agent in summary.agents:
            key = _agent_selection_key(agent, summary)
            current = best_agents.get(agent.id)
            if current is None or key < current[0]:
                best_agents[agent.id] = (key, agent)

    selected = sorted(best_agents.items(), key=lambda item: item[1][0])[:max_agents]
    return [agent_id for agent_id, _ in selected]


def _agent_selection_key(agent: AgentState, summary: SceneSummary) -> tuple[float, float, str]:
    distance = _agent_distance(agent, summary)
    risk_score = _agent_risk_score(agent, summary)
    effective_distance = distance / (1.0 + risk_score)
    return (effective_distance, distance, agent.id)


def _agent_distance(agent: AgentState, summary: SceneSummary) -> float:
    return math.hypot(agent.x - summary.ego.x, agent.y - summary.ego.y)


def _agent_risk_score(agent: AgentState, summary: SceneSummary) -> float:
    distance = _agent_distance(agent, summary)
    confidence_term = agent.confidence * agent.existence_prob
    size_term = max(0.0, agent.length * agent.width) / 20.0
    ambiguity_term = 0.5 if agent.ambiguity_flags else 0.0
    proximity_term = 1.0 / (1.0 + distance)
    return confidence_term + size_term + ambiguity_term + proximity_term


def _covariance_value(agent: AgentState, row: int, column: int) -> float:
    try:
        value = agent.covariance_xy[row][column]
    except (IndexError, TypeError) as error:
        raise ValueError("agent.covariance_xy must be at least 2x2 for tokenization") from error
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError("agent.covariance_xy values must be numeric")
    return float(value)
