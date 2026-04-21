"""Top-level minimal CoordiWorld model wrapper."""

from __future__ import annotations

from torch import Tensor, nn

from coordiworld.models.rollout import StructuredRolloutModel, StructuredRolloutOutput


class CoordiWorldModel(nn.Module):
    """Minimal structured rollout model entry point.

    This class only implements candidate-conditioned rollout. It does not
    include ranking, calibration, risk evaluation, or training loops.
    """

    def __init__(
        self,
        rollout_model: StructuredRolloutModel | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.rollout_model = rollout_model or StructuredRolloutModel(**kwargs)

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
        return self.rollout_model(
            ego_history=ego_history,
            agent_history=agent_history,
            map_history=map_history,
            action_tokens=action_tokens,
            agent_mask=agent_mask,
            map_mask=map_mask,
        )
