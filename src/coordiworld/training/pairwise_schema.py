"""Pairwise ranking batch schema for Stage II interface stability."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

PAIRWISE_RANKING_BATCH_FIELDS: tuple[str, ...] = (
    "ego_history",
    "agent_history",
    "map_history",
    "action_tokens",
    "agent_mask",
    "map_mask",
    "candidate_mask",
    "candidate_scores",
    "preferred_indices",
    "dispreferred_indices",
    "pairwise_margins",
    "candidate_metadata",
)


@dataclass(frozen=True)
class PairwiseRankingBatch:
    """Fixed Stage II batch contract for shared-candidate pairwise ranking."""

    ego_history: Tensor
    agent_history: Tensor
    map_history: Tensor
    action_tokens: Tensor
    agent_mask: Tensor
    map_mask: Tensor
    candidate_mask: Tensor
    candidate_scores: Tensor
    preferred_indices: Tensor
    dispreferred_indices: Tensor
    pairwise_margins: Tensor
    candidate_metadata: list[dict[str, object]]

    @property
    def batch_size(self) -> int:
        return int(self.action_tokens.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.action_tokens.shape[1])

    @property
    def horizon_steps(self) -> int:
        return int(self.action_tokens.shape[2])

    @property
    def pair_count(self) -> int:
        return int(self.preferred_indices.shape[1])

    def to(self, device: torch.device | str) -> "PairwiseRankingBatch":
        return PairwiseRankingBatch(
            ego_history=self.ego_history.to(device),
            agent_history=self.agent_history.to(device),
            map_history=self.map_history.to(device),
            action_tokens=self.action_tokens.to(device),
            agent_mask=self.agent_mask.to(device),
            map_mask=self.map_mask.to(device),
            candidate_mask=self.candidate_mask.to(device),
            candidate_scores=self.candidate_scores.to(device),
            preferred_indices=self.preferred_indices.to(device),
            dispreferred_indices=self.dispreferred_indices.to(device),
            pairwise_margins=self.pairwise_margins.to(device),
            candidate_metadata=self.candidate_metadata,
        )


def validate_pairwise_ranking_batch(batch: PairwiseRankingBatch) -> None:
    """Validate the fixed Stage II pairwise ranking batch schema."""
    if not isinstance(batch, PairwiseRankingBatch):
        raise ValueError("batch must be a PairwiseRankingBatch")
    _require_rank(batch.ego_history, 3, "ego_history")
    _require_rank(batch.agent_history, 4, "agent_history")
    _require_rank(batch.map_history, 4, "map_history")
    _require_rank(batch.action_tokens, 4, "action_tokens")
    _require_rank(batch.agent_mask, 3, "agent_mask")
    _require_rank(batch.map_mask, 3, "map_mask")
    _require_rank(batch.candidate_mask, 2, "candidate_mask")
    _require_rank(batch.candidate_scores, 2, "candidate_scores")
    _require_rank(batch.preferred_indices, 2, "preferred_indices")
    _require_rank(batch.dispreferred_indices, 2, "dispreferred_indices")
    _require_rank(batch.pairwise_margins, 2, "pairwise_margins")

    batch_size, history_length = batch.ego_history.shape[:2]
    candidate_count = batch.action_tokens.shape[1]
    if batch.agent_history.shape[:2] != (batch_size, history_length):
        raise ValueError("agent_history must share ego_history [B,T]")
    if batch.map_history.shape[:2] != (batch_size, history_length):
        raise ValueError("map_history must share ego_history [B,T]")
    if batch.agent_mask.shape != batch.agent_history.shape[:3]:
        raise ValueError("agent_mask must match agent_history [B,T,N]")
    if batch.map_mask.shape != batch.map_history.shape[:3]:
        raise ValueError("map_mask must match map_history [B,T,K]")
    if batch.candidate_mask.shape != (batch_size, candidate_count):
        raise ValueError("candidate_mask must match [B,M]")
    if batch.candidate_scores.shape != (batch_size, candidate_count):
        raise ValueError("candidate_scores must match [B,M]")

    pair_shape = batch.preferred_indices.shape
    if batch.dispreferred_indices.shape != pair_shape:
        raise ValueError("dispreferred_indices must match preferred_indices shape")
    if batch.pairwise_margins.shape != pair_shape:
        raise ValueError("pairwise_margins must match preferred_indices shape")
    if pair_shape[0] != batch_size:
        raise ValueError("pairwise index tensors must share batch size")
    if len(batch.candidate_metadata) != batch_size:
        raise ValueError("candidate_metadata length must match batch size")
    if not all(isinstance(item, dict) for item in batch.candidate_metadata):
        raise ValueError("candidate_metadata entries must be dicts")

    _require_finite(batch.ego_history, "ego_history")
    _require_finite(batch.agent_history, "agent_history")
    _require_finite(batch.map_history, "map_history")
    _require_finite(batch.action_tokens, "action_tokens")
    _require_finite(batch.candidate_scores, "candidate_scores")
    _require_finite(batch.pairwise_margins, "pairwise_margins")
    if torch.any(batch.pairwise_margins < 0):
        raise ValueError("pairwise_margins must be non-negative")

    _validate_pair_indices(batch.preferred_indices, candidate_count, "preferred_indices")
    _validate_pair_indices(batch.dispreferred_indices, candidate_count, "dispreferred_indices")
    if torch.any(batch.preferred_indices == batch.dispreferred_indices):
        raise ValueError("preferred and dispreferred indices must differ")


def _validate_pair_indices(indices: Tensor, candidate_count: int, name: str) -> None:
    if not torch.is_floating_point(indices) and indices.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        pass
    else:
        raise ValueError(f"{name} must be an integer tensor")
    if torch.any(indices < 0) or torch.any(indices >= candidate_count):
        raise ValueError(f"{name} must be within [0, M)")


def _require_rank(tensor: Tensor, rank: int, name: str) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got {tensor.ndim}")


def _require_finite(tensor: Tensor, name: str) -> None:
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must be finite")
