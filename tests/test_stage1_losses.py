"""Tests for Stage I structured rollout losses."""

from __future__ import annotations

import torch

from coordiworld.models.rollout import StructuredRolloutOutput
from coordiworld.training.losses import (
    compute_stage1_rollout_loss,
    existence_bce_loss,
    gaussian_nll_xy,
    huber_velocity_loss,
    huber_yaw_loss,
    risk_bce_loss,
)


def test_gaussian_nll_xy_prefers_accurate_predictions() -> None:
    target = torch.zeros(1, 1, 2, 1, 2)
    log_variance = torch.zeros_like(target)
    good = torch.zeros_like(target)
    bad = torch.ones_like(target) * 3.0

    assert gaussian_nll_xy(good, target, log_variance) < gaussian_nll_xy(
        bad,
        target,
        log_variance,
    )


def test_huber_yaw_and_velocity_losses_are_zero_for_exact_match() -> None:
    yaw = torch.tensor([0.1, -0.2])
    velocity = torch.tensor([[1.0, 0.5], [0.0, -1.0]])

    assert huber_yaw_loss(yaw, yaw).item() == 0.0
    assert huber_velocity_loss(velocity, velocity).item() == 0.0


def test_bce_losses_are_finite_and_lower_for_correct_logits() -> None:
    targets = torch.tensor([1.0, 0.0])
    correct_logits = torch.tensor([4.0, -4.0])
    wrong_logits = -correct_logits

    assert existence_bce_loss(correct_logits, targets) < existence_bce_loss(
        wrong_logits,
        targets,
    )
    assert risk_bce_loss(correct_logits, targets) < risk_bce_loss(wrong_logits, targets)


def test_compute_stage1_rollout_loss_backward() -> None:
    states = torch.zeros(1, 1, 2, 1, 5, requires_grad=True)
    existence_logits = torch.zeros(1, 1, 2, 1, requires_grad=True)
    log_variance = torch.zeros(1, 1, 2, 1, 2, requires_grad=True)
    sigma = torch.ones_like(log_variance)
    output = StructuredRolloutOutput(
        state_deltas=states,
        agent_states=states,
        existence_logits=existence_logits,
        covariance_log_variance=log_variance,
        covariance_sigma=sigma,
        scene_context=torch.zeros(1, 8),
    )
    target_states = torch.ones_like(states)
    target_existence = torch.ones_like(existence_logits)

    loss = compute_stage1_rollout_loss(
        output,
        target_agent_states=target_states,
        target_existence=target_existence,
        risk_logits=torch.zeros(1, 1, requires_grad=True),
        risk_labels=torch.ones(1, 1),
        mask=target_existence,
    )
    loss.total.backward()

    assert loss.total.item() > 0.0
    assert states.grad is not None
    assert existence_logits.grad is not None
    assert log_variance.grad is not None
