"""Tests for the fixed Stage II pairwise ranking batch schema."""

from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from coordiworld.training.pairwise_schema import (
    PAIRWISE_RANKING_BATCH_FIELDS,
    PairwiseRankingBatch,
    validate_pairwise_ranking_batch,
)


def make_batch() -> PairwiseRankingBatch:
    return PairwiseRankingBatch(
        ego_history=torch.zeros(2, 3, 7),
        agent_history=torch.zeros(2, 3, 32, 16),
        map_history=torch.zeros(2, 3, 24, 10),
        action_tokens=torch.zeros(2, 4, 5, 6),
        agent_mask=torch.ones(2, 3, 32),
        map_mask=torch.ones(2, 3, 24),
        candidate_mask=torch.ones(2, 4),
        candidate_scores=torch.tensor([[0.9, 0.6, 0.2, 0.1], [0.8, 0.4, 0.3, 0.0]]),
        preferred_indices=torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.long),
        dispreferred_indices=torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long),
        pairwise_margins=torch.tensor([[0.3, 0.7, 0.5], [0.4, 0.1, 0.3]]),
        candidate_metadata=[
            {"candidate_pool_type": "shared", "seed": 0},
            {"candidate_pool_type": "shared", "seed": 0},
        ],
    )


def test_pairwise_ranking_batch_field_contract_is_stable() -> None:
    assert tuple(field.name for field in fields(PairwiseRankingBatch)) == (
        PAIRWISE_RANKING_BATCH_FIELDS
    )


def test_valid_pairwise_ranking_batch_passes_validation() -> None:
    batch = make_batch()

    validate_pairwise_ranking_batch(batch)

    assert batch.batch_size == 2
    assert batch.candidate_count == 4
    assert batch.horizon_steps == 5
    assert batch.pair_count == 3


def test_pairwise_batch_to_preserves_metadata() -> None:
    batch = make_batch()

    moved = batch.to("cpu")

    validate_pairwise_ranking_batch(moved)
    assert moved.candidate_metadata == batch.candidate_metadata


def test_pairwise_batch_rejects_bad_candidate_score_shape() -> None:
    batch = replace(make_batch(), candidate_scores=torch.zeros(2, 3))

    with pytest.raises(ValueError, match=r"\[B,M\]"):
        validate_pairwise_ranking_batch(batch)


def test_pairwise_batch_rejects_invalid_pair_indices() -> None:
    batch = replace(make_batch(), preferred_indices=torch.tensor([[0, 4, 1], [0, 1, 2]]))

    with pytest.raises(ValueError, match=r"\[0, M\)"):
        validate_pairwise_ranking_batch(batch)


def test_pairwise_batch_rejects_equal_pair_indices() -> None:
    batch = replace(make_batch(), dispreferred_indices=make_batch().preferred_indices.clone())

    with pytest.raises(ValueError, match="must differ"):
        validate_pairwise_ranking_batch(batch)
