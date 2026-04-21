"""CPU smoke tests for the minimal structured rollout model."""

from __future__ import annotations

import math

import torch

from coordiworld.models.coordiworld import CoordiWorldModel
from coordiworld.models.rollout import StructuredRolloutModel, compose_residual_state


def make_inputs(
    *,
    batch_size: int = 2,
    history: int = 3,
    candidates: int = 4,
    horizon: int = 5,
    agents: int = 6,
    map_tokens: int = 8,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        "ego_history": torch.randn(batch_size, history, 7),
        "agent_history": torch.randn(batch_size, history, agents, 16),
        "map_history": torch.randn(batch_size, history, map_tokens, 10),
        "action_tokens": torch.randn(batch_size, candidates, horizon, 6),
        "agent_mask": torch.ones(batch_size, history, agents),
        "map_mask": torch.ones(batch_size, history, map_tokens),
    }


def test_structured_rollout_forward_shapes_on_cpu() -> None:
    inputs = make_inputs()
    model = StructuredRolloutModel(hidden_dim=32, num_heads=4, num_layers=1)

    output = model(**inputs)

    assert output.state_deltas.shape == (2, 4, 5, 6, 5)
    assert output.agent_states.shape == (2, 4, 5, 6, 5)
    assert output.existence_logits.shape == (2, 4, 5, 6)
    assert output.covariance_log_variance.shape == (2, 4, 5, 6, 2)
    assert output.covariance_sigma.shape == (2, 4, 5, 6, 2)
    assert output.scene_context.shape == (2, 32)
    assert torch.all(output.covariance_sigma > 0.0)


def test_action_conditioning_changes_rollout() -> None:
    inputs = make_inputs(batch_size=1, candidates=1, horizon=3, agents=2)
    model = StructuredRolloutModel(hidden_dim=32, num_heads=4, num_layers=1)
    model.eval()

    zero_actions = dict(inputs)
    one_actions = dict(inputs)
    zero_actions["action_tokens"] = torch.zeros_like(inputs["action_tokens"])
    one_actions["action_tokens"] = torch.ones_like(inputs["action_tokens"])

    with torch.no_grad():
        zero_output = model(**zero_actions)
        one_output = model(**one_actions)

    assert not torch.allclose(zero_output.state_deltas, one_output.state_deltas)


def test_residual_composition_wraps_yaw() -> None:
    current = torch.tensor([[[[1.0, 2.0, math.pi - 0.1, 0.5, -0.5]]]])
    delta = torch.tensor([[[[0.5, -1.0, 0.3, 0.2, 0.1]]]])

    composed = compose_residual_state(current, delta)

    assert composed.shape == current.shape
    assert torch.allclose(composed[..., 0], torch.tensor([[[1.5]]]))
    assert torch.allclose(composed[..., 1], torch.tensor([[[1.0]]]))
    assert composed[..., 2].item() < 0.0
    assert torch.allclose(composed[..., 3:], torch.tensor([[[[0.7, -0.4]]]]))


def test_masked_agents_remain_zeroed() -> None:
    inputs = make_inputs(batch_size=1, candidates=2, horizon=3, agents=3)
    inputs["agent_mask"][:, :, -1] = 0.0
    model = StructuredRolloutModel(hidden_dim=32, num_heads=4, num_layers=1)

    output = model(**inputs)

    assert torch.all(output.state_deltas[:, :, :, -1, :] == 0.0)
    assert torch.all(output.agent_states[:, :, :, -1, :] == 0.0)
    assert torch.all(output.existence_logits[:, :, :, -1] < -1000.0)


def test_model_is_trainable_with_backward_pass() -> None:
    inputs = make_inputs(batch_size=1, candidates=2, horizon=3, agents=4)
    model = CoordiWorldModel(hidden_dim=32, num_heads=4, num_layers=1)

    output = model(**inputs)
    loss = (
        output.agent_states.square().mean()
        + output.existence_logits.square().mean()
        + output.covariance_log_variance.square().mean()
    )
    loss.backward()

    grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert grad_norm > 0.0
