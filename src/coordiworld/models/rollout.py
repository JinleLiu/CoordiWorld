"""Candidate-conditioned structured rollout model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from coordiworld.models.encoder import TemporalSceneEncoder
from coordiworld.models.heads import AgentRolloutHead


@dataclass(frozen=True)
class StructuredRolloutOutput:
    state_deltas: Tensor
    agent_states: Tensor
    existence_logits: Tensor
    covariance_log_variance: Tensor
    covariance_sigma: Tensor
    scene_context: Tensor


class StructuredRolloutModel(nn.Module):
    """Minimal trainable rollout model for fixed candidate-set evaluation."""

    def __init__(
        self,
        *,
        ego_dim: int = 7,
        agent_dim: int = 16,
        map_dim: int = 10,
        action_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = TemporalSceneEncoder(
            ego_dim=ego_dim,
            agent_dim=agent_dim,
            map_dim=map_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.rollout_head = AgentRolloutHead(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            state_dim=5,
            residual_dim=5,
        )

    def forward(
        self,
        *,
        ego_history: Tensor,
        agent_history: Tensor,
        map_history: Tensor,
        action_tokens: Tensor,
        agent_mask: Tensor | None = None,
        map_mask: Tensor | None = None,
    ) -> StructuredRolloutOutput:
        """Forward pass.

        Shapes:
        - ego_history: [B,T,7]
        - agent_history: [B,T,N,16]
        - map_history: [B,T,K,10] or [B,K,10]
        - action_tokens: [B,M,H,6]
        - output tensors: [B,M,H,N,...]
        """
        if action_tokens.ndim != 4:
            raise ValueError("action_tokens must have shape [B,M,H,U]")
        if agent_history.ndim != 4:
            raise ValueError("agent_history must have shape [B,T,N,A]")
        if action_tokens.shape[0] != agent_history.shape[0]:
            raise ValueError("action_tokens and agent_history must share batch size")

        encoding = self.encoder(
            ego_history,
            agent_history,
            map_history,
            agent_mask=agent_mask,
            map_mask=map_mask,
        )

        batch, candidate_count, horizon, _ = action_tokens.shape
        latest_agent_state = agent_history[:, -1, :, :5]
        current_state = latest_agent_state[:, None, :, :].expand(
            batch,
            candidate_count,
            latest_agent_state.shape[1],
            5,
        )
        latest_agent_mask = _latest_agent_mask(agent_mask, latest_agent_state)

        state_deltas: list[Tensor] = []
        agent_states: list[Tensor] = []
        existence_logits: list[Tensor] = []
        covariance_log_variance: list[Tensor] = []
        covariance_sigma: list[Tensor] = []

        for step_index in range(horizon):
            head_output = self.rollout_head(
                encoding.context,
                action_tokens[:, :, step_index, :],
                current_state,
            )
            next_state = compose_residual_state(current_state, head_output.state_deltas)
            if latest_agent_mask is not None:
                mask = latest_agent_mask[:, None, :, None]
                next_state = next_state * mask
                step_deltas = head_output.state_deltas * mask
                step_sigma = (
                    head_output.covariance_sigma * mask
                    + (1.0 - mask) * self.rollout_head.min_sigma
                )
                step_log_variance = 2.0 * torch.log(step_sigma)
                step_existence = head_output.existence_logits.masked_fill(
                    latest_agent_mask[:, None, :] == 0,
                    -1.0e4,
                )
            else:
                step_deltas = head_output.state_deltas
                step_sigma = head_output.covariance_sigma
                step_log_variance = head_output.covariance_log_variance
                step_existence = head_output.existence_logits

            state_deltas.append(step_deltas)
            agent_states.append(next_state)
            existence_logits.append(step_existence)
            covariance_sigma.append(step_sigma)
            covariance_log_variance.append(step_log_variance)
            current_state = next_state

        return StructuredRolloutOutput(
            state_deltas=torch.stack(state_deltas, dim=2),
            agent_states=torch.stack(agent_states, dim=2),
            existence_logits=torch.stack(existence_logits, dim=2),
            covariance_log_variance=torch.stack(covariance_log_variance, dim=2),
            covariance_sigma=torch.stack(covariance_sigma, dim=2),
            scene_context=encoding.context,
        )


def compose_residual_state(current_state: Tensor, state_delta: Tensor) -> Tensor:
    """Apply residual composition to [x,y,yaw,vx,vy] agent state."""
    if current_state.shape != state_delta.shape:
        raise ValueError("current_state and state_delta must have identical shape")
    next_state = current_state + state_delta
    next_yaw = torch.atan2(torch.sin(next_state[..., 2]), torch.cos(next_state[..., 2]))
    return torch.stack(
        [
            next_state[..., 0],
            next_state[..., 1],
            next_yaw,
            next_state[..., 3],
            next_state[..., 4],
        ],
        dim=-1,
    )


def _latest_agent_mask(agent_mask: Tensor | None, latest_agent_state: Tensor) -> Tensor | None:
    if agent_mask is None:
        return None
    if agent_mask.ndim == 3:
        mask = agent_mask[:, -1, :]
    elif agent_mask.ndim == 2:
        mask = agent_mask
    else:
        raise ValueError("agent_mask must have shape [B,T,N] or [B,N]")
    if mask.shape != latest_agent_state.shape[:2]:
        raise ValueError("agent_mask latest shape must match [B,N]")
    return mask.to(device=latest_agent_state.device, dtype=latest_agent_state.dtype)
