"""Synthetic one-batch smoke tests for Stage I rollout pretraining."""

from __future__ import annotations

import torch

from coordiworld.data.candidate_pool import CandidatePoolConfig
from coordiworld.data.synthetic import SyntheticDatasetConfig, SyntheticScenarioDataset
from coordiworld.models.coordiworld import CoordiWorldModel
from coordiworld.training.stage1_rollout import build_stage1_batch
from coordiworld.training.trainer import Stage1RolloutTrainer, evaluate_stage1_loss


def make_samples():
    dataset = SyntheticScenarioDataset(
        SyntheticDatasetConfig(
            num_samples=2,
            history_length=2,
            candidate_pool_config=CandidatePoolConfig(horizon_steps=3),
        )
    )
    return [dataset[0], dataset[1]]


def test_build_stage1_batch_uses_logged_ego_future_as_action_token() -> None:
    samples = make_samples()
    batch = build_stage1_batch(samples)

    assert batch.ego_history.shape == (2, 2, 7)
    assert batch.agent_history.shape == (2, 2, 32, 16)
    assert batch.map_history.shape == (2, 2, 24, 10)
    assert batch.action_tokens.shape == (2, 1, 3, 6)
    assert batch.action_tokens[0, 0, 0, 0].item() == samples[0].logged_ego_future[0][0]
    assert batch.target_agent_states.shape == (2, 1, 3, 32, 5)
    assert batch.target_existence.sum().item() == 6.0


def test_one_batch_synthetic_smoke_train_updates_parameters() -> None:
    torch.manual_seed(11)
    samples = make_samples()
    model = CoordiWorldModel(hidden_dim=32, num_heads=4, num_layers=1)
    trainer = Stage1RolloutTrainer(model)
    batch = build_stage1_batch(samples)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    result = trainer.train_one_batch(batch)

    after = list(model.parameters())
    assert result.loss_value > 0.0
    assert result.components["xy_nll"] >= 0.0
    assert any(not torch.allclose(old, new) for old, new in zip(before, after))


def test_evaluate_stage1_loss_is_non_mutating() -> None:
    samples = make_samples()
    model = CoordiWorldModel(hidden_dim=32, num_heads=4, num_layers=1)
    batch = build_stage1_batch(samples)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = evaluate_stage1_loss(model, batch)

    assert loss.total.item() > 0.0
    assert all(torch.allclose(old, new) for old, new in zip(before, model.parameters()))
