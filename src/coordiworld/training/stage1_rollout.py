"""Stage I structured rollout pretraining batch utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from coordiworld.data.base import BaseScenarioSample
from coordiworld.tokens.action_tokenizer import ActionTokenizer
from coordiworld.tokens.scene_tokenizer import SceneTokenizer


@dataclass(frozen=True)
class Stage1Batch:
    ego_history: Tensor
    agent_history: Tensor
    map_history: Tensor
    action_tokens: Tensor
    agent_mask: Tensor
    map_mask: Tensor
    target_agent_states: Tensor
    target_existence: Tensor
    risk_labels: Tensor

    def to(self, device: torch.device | str) -> "Stage1Batch":
        return Stage1Batch(
            ego_history=self.ego_history.to(device),
            agent_history=self.agent_history.to(device),
            map_history=self.map_history.to(device),
            action_tokens=self.action_tokens.to(device),
            agent_mask=self.agent_mask.to(device),
            map_mask=self.map_mask.to(device),
            target_agent_states=self.target_agent_states.to(device),
            target_existence=self.target_existence.to(device),
            risk_labels=self.risk_labels.to(device),
        )


def build_stage1_batch(
    samples: Sequence[BaseScenarioSample],
    *,
    scene_tokenizer: SceneTokenizer | None = None,
    action_tokenizer: ActionTokenizer | None = None,
    device: torch.device | str = "cpu",
) -> Stage1Batch:
    """Build a Stage I batch using logged ego future as the sole action candidate."""
    if not samples:
        raise ValueError("samples must not be empty")
    resolved_scene_tokenizer = scene_tokenizer or SceneTokenizer()
    resolved_action_tokenizer = action_tokenizer or ActionTokenizer()

    encoded_samples = [
        _encode_sample(sample, resolved_scene_tokenizer, resolved_action_tokenizer)
        for sample in samples
    ]
    history_length = len(encoded_samples[0]["ego_history"])
    horizon = len(encoded_samples[0]["action_tokens"][0])
    for index, encoded in enumerate(encoded_samples):
        if len(encoded["ego_history"]) != history_length:
            raise ValueError(f"samples[{index}] history length mismatch")
        if len(encoded["action_tokens"][0]) != horizon:
            raise ValueError(f"samples[{index}] future horizon mismatch")

    batch = Stage1Batch(
        ego_history=_tensor([encoded["ego_history"] for encoded in encoded_samples]),
        agent_history=_tensor([encoded["agent_history"] for encoded in encoded_samples]),
        map_history=_tensor([encoded["map_history"] for encoded in encoded_samples]),
        action_tokens=_tensor([encoded["action_tokens"] for encoded in encoded_samples]),
        agent_mask=_tensor([encoded["agent_mask"] for encoded in encoded_samples]),
        map_mask=_tensor([encoded["map_mask"] for encoded in encoded_samples]),
        target_agent_states=_tensor(
            [encoded["target_agent_states"] for encoded in encoded_samples]
        ),
        target_existence=_tensor([encoded["target_existence"] for encoded in encoded_samples]),
        risk_labels=_tensor([encoded["risk_labels"] for encoded in encoded_samples]),
    )
    return batch.to(device)


def _encode_sample(
    sample: BaseScenarioSample,
    scene_tokenizer: SceneTokenizer,
    action_tokenizer: ActionTokenizer,
) -> dict[str, object]:
    clip_tokens = scene_tokenizer.tokenize(sample.scene_summary_history)
    frame_tokens = [scene_tokenizer.tokenize(summary) for summary in sample.scene_summary_history]
    action_tokens = action_tokenizer.tokenize([sample.logged_ego_future]).action_tensor
    target_states, target_existence = _build_future_targets(sample, clip_tokens.local_slot_indices)
    risk_label = 1.0 if sample.labels.collision or sample.labels.violation else 0.0
    return {
        "ego_history": [tokens.ego_tensor for tokens in frame_tokens],
        "agent_history": [tokens.agent_tensor for tokens in frame_tokens],
        "map_history": [tokens.map_tensor for tokens in frame_tokens],
        "action_tokens": action_tokens,
        "agent_mask": [tokens.masks["agents"] for tokens in frame_tokens],
        "map_mask": [tokens.masks["map_tokens"] for tokens in frame_tokens],
        "target_agent_states": target_states,
        "target_existence": target_existence,
        "risk_labels": [risk_label],
    }


def _build_future_targets(
    sample: BaseScenarioSample,
    local_slot_indices: dict[str, int],
) -> tuple[list[list[list[list[float]]]], list[list[list[float]]]]:
    horizon = len(sample.logged_ego_future)
    agent_count = 32
    target_states = [
        [
            [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(agent_count)]
            for _ in range(horizon)
        ]
    ]
    target_existence = [[[0.0 for _ in range(agent_count)] for _ in range(horizon)]]
    if not sample.future_agents:
        return target_states, target_existence

    for agent_id, trajectory in sample.future_agents.items():
        slot = local_slot_indices.get(agent_id)
        if slot is None:
            continue
        latest_agent = _latest_agent_by_id(sample, agent_id)
        previous_x = latest_agent[0] if latest_agent is not None else trajectory[0][0]
        previous_y = latest_agent[1] if latest_agent is not None else trajectory[0][1]
        for step_index, pose in enumerate(trajectory):
            x = float(pose[0])
            y = float(pose[1])
            yaw = float(pose[2])
            vx = x - previous_x
            vy = y - previous_y
            target_states[0][step_index][slot] = [x, y, yaw, vx, vy]
            target_existence[0][step_index][slot] = 1.0
            previous_x = x
            previous_y = y
    return target_states, target_existence


def _latest_agent_by_id(sample: BaseScenarioSample, agent_id: str) -> tuple[float, float] | None:
    for agent in sample.scene_summary_history[-1].agents:
        if agent.id == agent_id:
            return agent.x, agent.y
    return None


def _tensor(values: object) -> Tensor:
    return torch.tensor(values, dtype=torch.float32)
