"""Output heads for action-conditioned structured rollout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RolloutHeadOutput:
    state_deltas: Tensor
    existence_logits: Tensor
    covariance_log_variance: Tensor
    covariance_sigma: Tensor


class AgentRolloutHead(nn.Module):
    """Predict per-agent residual dynamics conditioned on candidate action tokens."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        action_dim: int = 6,
        state_dim: int = 5,
        residual_dim: int = 5,
        min_sigma: float = 1e-3,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.residual_dim = residual_dim
        self.min_sigma = min_sigma
        self.residual_scale = residual_scale
        input_dim = hidden_dim + action_dim + state_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_dim, residual_dim)
        self.existence_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        context: Tensor,
        action_tokens: Tensor,
        current_state: Tensor,
    ) -> RolloutHeadOutput:
        """Run one rollout step.

        Shapes:
        - context: [B,D]
        - action_tokens: [B,M,U]
        - current_state: [B,M,N,5]
        """
        if context.ndim != 2:
            raise ValueError("context must have shape [B,D]")
        if action_tokens.ndim != 3:
            raise ValueError("action_tokens must have shape [B,M,U]")
        if current_state.ndim != 4:
            raise ValueError("current_state must have shape [B,M,N,5]")

        batch, candidate_count, agent_count, _ = current_state.shape
        context_expanded = context[:, None, None, :].expand(
            batch,
            candidate_count,
            agent_count,
            context.shape[-1],
        )
        action_expanded = action_tokens[:, :, None, :].expand(
            batch,
            candidate_count,
            agent_count,
            action_tokens.shape[-1],
        )
        features = torch.cat([context_expanded, action_expanded, current_state], dim=-1)
        hidden = self.network(features)
        state_deltas = self.residual_scale * torch.tanh(self.delta_head(hidden))
        existence_logits = self.existence_head(hidden).squeeze(-1)
        sigma = F.softplus(self.sigma_head(hidden)) + self.min_sigma
        log_variance = 2.0 * torch.log(sigma)
        return RolloutHeadOutput(
            state_deltas=state_deltas,
            existence_logits=existence_logits,
            covariance_log_variance=log_variance,
            covariance_sigma=sigma,
        )
