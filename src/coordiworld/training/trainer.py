"""Minimal Stage I trainer utilities for synthetic smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from coordiworld.data.base import BaseScenarioSample
from coordiworld.models.coordiworld import CoordiWorldModel
from coordiworld.training.losses import (
    Stage1LossOutput,
    Stage1LossWeights,
    compute_stage1_rollout_loss,
)
from coordiworld.training.stage1_rollout import Stage1Batch, build_stage1_batch


@dataclass(frozen=True)
class Stage1TrainStepResult:
    loss_value: float
    components: dict[str, float]


class Stage1RolloutTrainer:
    """Tiny trainer wrapper for one-batch structured rollout pretraining."""

    def __init__(
        self,
        model: CoordiWorldModel,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        loss_weights: Stage1LossWeights | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_weights = loss_weights or Stage1LossWeights()

    def train_one_batch(self, batch: Stage1Batch) -> Stage1TrainStepResult:
        self.model.train()
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(
            ego_history=batch.ego_history,
            agent_history=batch.agent_history,
            map_history=batch.map_history,
            action_tokens=batch.action_tokens,
            agent_mask=batch.agent_mask,
            map_mask=batch.map_mask,
        )
        risk_logits = output.existence_logits.mean(dim=(2, 3))
        loss_output = compute_stage1_rollout_loss(
            output,
            target_agent_states=batch.target_agent_states,
            target_existence=batch.target_existence,
            risk_logits=risk_logits,
            risk_labels=batch.risk_labels,
            mask=batch.target_existence,
            weights=self.loss_weights,
        )
        loss_output.total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        return Stage1TrainStepResult(
            loss_value=float(loss_output.total.detach().cpu()),
            components=loss_output.components(),
        )

    def train_one_synthetic_batch(
        self,
        samples: Sequence[BaseScenarioSample],
    ) -> Stage1TrainStepResult:
        return self.train_one_batch(build_stage1_batch(samples, device=self.device))


def evaluate_stage1_loss(
    model: CoordiWorldModel,
    batch: Stage1Batch,
    *,
    loss_weights: Stage1LossWeights | None = None,
    device: torch.device | str = "cpu",
) -> Stage1LossOutput:
    """Compute Stage I loss without optimizer updates."""
    resolved_device = torch.device(device)
    model = model.to(resolved_device)
    model.eval()
    batch = batch.to(resolved_device)
    with torch.no_grad():
        output = model(
            ego_history=batch.ego_history,
            agent_history=batch.agent_history,
            map_history=batch.map_history,
            action_tokens=batch.action_tokens,
            agent_mask=batch.agent_mask,
            map_mask=batch.map_mask,
        )
        risk_logits = output.existence_logits.mean(dim=(2, 3))
        return compute_stage1_rollout_loss(
            output,
            target_agent_states=batch.target_agent_states,
            target_existence=batch.target_existence,
            risk_logits=risk_logits,
            risk_labels=batch.risk_labels,
            mask=batch.target_existence,
            weights=loss_weights,
        )
