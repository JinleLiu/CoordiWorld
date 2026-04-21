"""Stage I structured rollout pretraining losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from coordiworld.models.rollout import StructuredRolloutOutput


@dataclass(frozen=True)
class Stage1LossWeights:
    xy_nll: float = 1.0
    yaw_huber: float = 0.5
    velocity_huber: float = 0.5
    existence_bce: float = 1.0
    risk_bce: float = 0.25


@dataclass(frozen=True)
class Stage1LossOutput:
    total: Tensor
    xy_nll: Tensor
    yaw_huber: Tensor
    velocity_huber: Tensor
    existence_bce: Tensor
    risk_bce: Tensor

    def components(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "xy_nll": float(self.xy_nll.detach().cpu()),
            "yaw_huber": float(self.yaw_huber.detach().cpu()),
            "velocity_huber": float(self.velocity_huber.detach().cpu()),
            "existence_bce": float(self.existence_bce.detach().cpu()),
            "risk_bce": float(self.risk_bce.detach().cpu()),
        }


def gaussian_nll_xy(
    pred_xy: Tensor,
    target_xy: Tensor,
    log_variance_xy: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Gaussian NLL for planar xy predictions with diagonal log variance."""
    if pred_xy.shape != target_xy.shape:
        raise ValueError("pred_xy and target_xy must have identical shape")
    if pred_xy.shape != log_variance_xy.shape:
        raise ValueError("log_variance_xy must match pred_xy shape")
    squared_error = (target_xy - pred_xy).square()
    nll_per_dim = 0.5 * (log_variance_xy + squared_error * torch.exp(-log_variance_xy))
    return masked_mean(nll_per_dim.sum(dim=-1), mask)


def huber_yaw_loss(pred_yaw: Tensor, target_yaw: Tensor, *, mask: Tensor | None = None) -> Tensor:
    """Huber loss over wrapped yaw residuals."""
    if pred_yaw.shape != target_yaw.shape:
        raise ValueError("pred_yaw and target_yaw must have identical shape")
    residual = wrap_angle(pred_yaw - target_yaw)
    loss = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none")
    return masked_mean(loss, mask)


def huber_velocity_loss(
    pred_velocity: Tensor,
    target_velocity: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Huber loss for vx/vy."""
    if pred_velocity.shape != target_velocity.shape:
        raise ValueError("pred_velocity and target_velocity must have identical shape")
    loss = F.smooth_l1_loss(pred_velocity, target_velocity, reduction="none").sum(dim=-1)
    return masked_mean(loss, mask)


def existence_bce_loss(logits: Tensor, targets: Tensor, *, mask: Tensor | None = None) -> Tensor:
    """Binary cross entropy for agent existence logits."""
    if logits.shape != targets.shape:
        raise ValueError("existence logits and targets must have identical shape")
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return masked_mean(loss, mask)


def risk_bce_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Binary cross entropy for sample/candidate-level risk labels."""
    if logits.shape != targets.shape:
        raise ValueError("risk logits and targets must have identical shape")
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def compute_stage1_rollout_loss(
    output: StructuredRolloutOutput,
    *,
    target_agent_states: Tensor,
    target_existence: Tensor,
    risk_logits: Tensor | None = None,
    risk_labels: Tensor | None = None,
    mask: Tensor | None = None,
    weights: Stage1LossWeights | None = None,
) -> Stage1LossOutput:
    """Compute weighted Stage I rollout pretraining loss."""
    resolved_weights = weights or Stage1LossWeights()
    if output.agent_states.shape != target_agent_states.shape:
        raise ValueError("target_agent_states must match output.agent_states")
    if output.existence_logits.shape != target_existence.shape:
        raise ValueError("target_existence must match output.existence_logits")

    xy_nll = gaussian_nll_xy(
        output.agent_states[..., :2],
        target_agent_states[..., :2],
        output.covariance_log_variance,
        mask=mask,
    )
    yaw_huber = huber_yaw_loss(output.agent_states[..., 2], target_agent_states[..., 2], mask=mask)
    velocity_huber = huber_velocity_loss(
        output.agent_states[..., 3:5],
        target_agent_states[..., 3:5],
        mask=mask,
    )
    existence_loss = existence_bce_loss(output.existence_logits, target_existence, mask=mask)
    if risk_logits is None or risk_labels is None:
        risk_loss = output.agent_states.new_zeros(())
    else:
        risk_loss = risk_bce_loss(risk_logits, risk_labels)

    total = (
        resolved_weights.xy_nll * xy_nll
        + resolved_weights.yaw_huber * yaw_huber
        + resolved_weights.velocity_huber * velocity_huber
        + resolved_weights.existence_bce * existence_loss
        + resolved_weights.risk_bce * risk_loss
    )
    return Stage1LossOutput(
        total=total,
        xy_nll=xy_nll,
        yaw_huber=yaw_huber,
        velocity_huber=velocity_huber,
        existence_bce=existence_loss,
        risk_bce=risk_loss,
    )


def masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    """Mean over values with an optional broadcast-compatible binary mask."""
    if mask is None:
        return values.mean()
    resolved_mask = mask.to(device=values.device, dtype=values.dtype)
    while resolved_mask.ndim < values.ndim:
        resolved_mask = resolved_mask.unsqueeze(-1)
    weighted = values * resolved_mask
    return weighted.sum() / resolved_mask.sum().clamp_min(1.0)


def wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))
