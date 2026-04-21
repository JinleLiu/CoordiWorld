"""Temporal scene encoder for structured CoordiWorld tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SceneEncoding:
    temporal_tokens: Tensor
    context: Tensor


class TemporalSceneEncoder(nn.Module):
    """Encode scene history tokens with a lightweight Transformer encoder."""

    def __init__(
        self,
        *,
        ego_dim: int = 7,
        agent_dim: int = 16,
        map_dim: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.ego_projection = nn.Linear(ego_dim, hidden_dim)
        self.agent_projection = nn.Linear(agent_dim, hidden_dim)
        self.map_projection = nn.Linear(map_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        ego_history: Tensor,
        agent_history: Tensor,
        map_history: Tensor,
        *,
        agent_mask: Tensor | None = None,
        map_mask: Tensor | None = None,
    ) -> SceneEncoding:
        """Encode scene history.

        Shapes:
        - ego_history: [B,T,E]
        - agent_history: [B,T,N,A]
        - map_history: [B,T,K,P] or [B,K,P]
        """
        _require_rank(ego_history, 3, "ego_history")
        _require_rank(agent_history, 4, "agent_history")
        if ego_history.shape[:2] != agent_history.shape[:2]:
            raise ValueError("ego_history and agent_history must share [B,T]")

        map_history = _expand_map_history(map_history, ego_history.shape[1])
        if ego_history.shape[:2] != map_history.shape[:2]:
            raise ValueError("ego_history and map_history must share [B,T]")

        agent_summary = masked_mean(agent_history, agent_mask, dim=2)
        map_mask = _expand_map_mask(map_mask, map_history)
        map_summary = masked_mean(map_history, map_mask, dim=2)

        temporal_tokens = (
            self.ego_projection(ego_history)
            + self.agent_projection(agent_summary)
            + self.map_projection(map_summary)
        )
        temporal_tokens = temporal_tokens + sinusoidal_positions(
            length=temporal_tokens.shape[1],
            hidden_dim=temporal_tokens.shape[2],
            device=temporal_tokens.device,
            dtype=temporal_tokens.dtype,
        )
        encoded = self.temporal_encoder(temporal_tokens)
        encoded = self.output_norm(encoded)
        return SceneEncoding(temporal_tokens=encoded, context=encoded[:, -1, :])


def masked_mean(values: Tensor, mask: Tensor | None, *, dim: int) -> Tensor:
    """Mean-pool values with optional binary mask."""
    if mask is None:
        return values.mean(dim=dim)
    expanded_mask = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    summed = (values * expanded_mask).sum(dim=dim)
    counts = expanded_mask.sum(dim=dim).clamp_min(1.0)
    return summed / counts


def sinusoidal_positions(
    *,
    length: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create deterministic sinusoidal temporal positions with shape [1,T,D]."""
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    even_indices = torch.arange(0, hidden_dim, 2, device=device, dtype=dtype)
    div_term = torch.exp(
        even_indices * (-torch.log(torch.tensor(10000.0, device=device)) / hidden_dim)
    )
    embeddings = torch.zeros(length, hidden_dim, device=device, dtype=dtype)
    embeddings[:, 0::2] = torch.sin(positions * div_term)
    if hidden_dim > 1:
        embeddings[:, 1::2] = torch.cos(positions * div_term[: embeddings[:, 1::2].shape[1]])
    return embeddings.unsqueeze(0)


def _expand_map_history(map_history: Tensor, history_length: int) -> Tensor:
    if map_history.ndim == 3:
        return map_history.unsqueeze(1).expand(-1, history_length, -1, -1)
    _require_rank(map_history, 4, "map_history")
    return map_history


def _expand_map_mask(mask: Tensor | None, map_history: Tensor) -> Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 2:
        return mask.unsqueeze(1).expand(-1, map_history.shape[1], -1)
    _require_rank(mask, 3, "map_mask")
    return mask


def _require_rank(tensor: Tensor, rank: int, name: str) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got {tensor.ndim}")
