"""Revision V5: factual anchor, centered action response and isolated scoring.

This is a local research implementation, not an official benchmark adapter.
``executed_candidates`` must be a controller prediction computed at decision
time from current observations. Observed future execution and labels must never
be passed to this module. They belong exclusively in ``losses.py``.
"""
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from ..geometry import rear_axle_to_box, sat_clearance, wrap_angle
from ..model import CoordiWorld, ModelConfig, map_risk, mlp, normalized_state
from ..schema import SceneInputs


@dataclass
class RevisionV5Config:
    hidden_dim: int = 128
    heads: int = 4
    history_layers: int = 2
    feedforward_dim: int = 512
    dropout: float = 0.0
    max_history: int = 8
    max_agents: int = 64
    dt: float = 0.5
    freeze_anchor: bool = True
    detach_world_for_risk: bool = True
    action_conditioned: bool = True
    use_provenance: bool = True
    use_source: bool = True
    use_lineage: bool = True
    use_ambiguity: bool = True
    use_confidence: bool = True
    response_position_scale: float = 5.0
    response_yaw_scale: float = 0.4
    rule_temperature: float = 0.5
    violation_beta: float = 8.0
    # current_only is the NoWorld control, not a stationary-motion forecast.
    # It retains current/history observations and ego execution but never calls
    # the factual rollout or response network. Event/value require retraining.
    risk_world_mode: str = "predicted"

    def legacy_config(self) -> ModelConfig:
        return ModelConfig(
            hidden_dim=self.hidden_dim, heads=self.heads,
            history_layers=self.history_layers,
            feedforward_dim=self.feedforward_dim, dropout=self.dropout,
            max_history=self.max_history, max_agents=self.max_agents, dt=self.dt,
            use_provenance=self.use_provenance,
            use_confidence=self.use_confidence, action_conditioned=False,
        )


class StructuredHistoryEncoder(CoordiWorld):
    """Reuse the existing structured encoder without its unused scoring heads."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        for name in ("action_embedding", "recurrent", "motion_head", "collision_head"):
            delattr(self, name)

    def forward(self, scene: SceneInputs):
        return self.encode(scene)


def _masked_mean(values: Tensor, valid: Tensor, dim: int) -> Tensor:
    weights = valid.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim) / weights.sum(dim).clamp_min(1.0)


def _reference_indices(reference_action, valid: Tensor) -> Tensor:
    b, m = valid.shape
    if isinstance(reference_action, int):
        reference_action = torch.full((b,), reference_action, device=valid.device,
                                      dtype=torch.long)
    if not isinstance(reference_action, Tensor) or reference_action.shape != (b,):
        raise ValueError("reference_action must be an integer or a [B] tensor")
    if reference_action.dtype != torch.long or reference_action.device != valid.device:
        raise ValueError("reference_action must be a long tensor on the input device")
    if ((reference_action < 0) | (reference_action >= m)).any():
        raise ValueError("reference_action index is out of range")
    if not valid.gather(1, reference_action[:, None]).all():
        raise ValueError("reference_action must identify a valid candidate")
    return reference_action


def _finite_when(value: Tensor, valid: Tensor, name: str) -> None:
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    if not torch.isfinite(value.masked_select(valid.expand_as(value))).all():
        raise ValueError(f"{name} has non-finite values in an available entry")


def _clean_scene(scene: SceneInputs, executed: Tensor,
                 executed_rule_margins: Tensor | None,
                 executed_rule_valid: Tensor | None) -> SceneInputs:
    """Validate available observations and eliminate all padded token contents."""
    b, l, n, channels = scene.agents.shape
    if channels != 7 or scene.ego.shape != (b, l, 7):
        raise ValueError("scene history must use the SceneInputs seven-channel state")
    if executed.shape != scene.candidates.shape or executed.ndim != 4 or executed.shape[-1] != 3:
        raise ValueError("executed_candidates must match candidates [B,M,H,3]")
    m, h = executed.shape[1:3]
    p = scene.map_features.shape[1]
    expected = {
        "agent_valid": (b, l, n), "agent_class": (b, n),
        "confidence": (b, l, n, 3), "confidence_valid": (b, l, n, 3),
        "source": (b, l, n, 6), "lineage": (b, l, n),
        "ambiguity": (b, l, n), "map_features": (b, p, 8),
        "map_valid": (b, p), "map_kind": (b, p), "map_source": (b, p, 6),
        "candidate_valid": (b, m), "ego_rear_axle_to_center": (b,),
    }
    for name, shape in expected.items():
        if getattr(scene, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for name in ("agent_valid", "confidence_valid", "map_valid", "candidate_valid"):
        if getattr(scene, name).dtype != torch.bool:
            raise ValueError(f"{name} must have bool dtype")
    if h == 0 or not scene.candidate_valid.any(-1).all():
        raise ValueError("every scene needs a horizon and at least one valid candidate")
    if (executed_rule_margins is None) != (executed_rule_valid is None):
        raise ValueError("provide executed_rule_margins and executed_rule_valid together")
    if executed_rule_margins is None:
        compared = scene.candidate_valid[..., None, None].expand_as(executed)
        if not torch.equal(executed[compared], scene.candidates[compared]):
            raise ValueError("changed execution requires recomputed executed rule geometry")
        margins, rules = scene.rule_margins, scene.rule_valid
    else:
        margins, rules = executed_rule_margins, executed_rule_valid
    if margins.shape != (b, m, h, p, 4) or rules.shape != margins.shape:
        raise ValueError("executed rule geometry must be [B,M,H,P,4]")
    if rules.dtype != torch.bool:
        raise ValueError("executed_rule_valid must have bool dtype")
    av, mv = scene.agent_valid, scene.map_valid
    cv = scene.confidence_valid & av[..., None]
    rv = rules & mv[:, None, None, :, None] & scene.candidate_valid[:, :, None, None, None]
    _finite_when(scene.ego, torch.ones((b, l), dtype=torch.bool, device=scene.ego.device), "ego")
    _finite_when(scene.agents, av, "agents")
    _finite_when(scene.confidence, cv, "confidence")
    _finite_when(scene.source, av, "source")
    _finite_when(scene.map_features, mv, "map_features")
    _finite_when(scene.map_source, mv, "map_source")
    _finite_when(scene.candidates, scene.candidate_valid, "planned_candidates")
    _finite_when(executed, scene.candidate_valid, "executed_candidates")
    _finite_when(margins, rv, "executed_rule_margins")
    if not torch.isfinite(scene.ego_rear_axle_to_center).all():
        raise ValueError("ego_rear_axle_to_center must be finite")
    if (scene.ego[..., 5:7] <= 0).any() or (scene.agents[..., 5:7][av] <= 0).any():
        raise ValueError("available ego/actor dimensions must be positive")
    if ((scene.confidence[cv] < 0) | (scene.confidence[cv] > 1)).any():
        raise ValueError("available confidence must be in [0,1]")
    if ((scene.ambiguity[av] != 0) & (scene.ambiguity[av] != 1)).any():
        raise ValueError("available ambiguity must be binary")
    for value, mask, upper, name in (
        (scene.agent_class, av.any(1), 16, "agent_class"),
        (scene.lineage, av, 3, "lineage"),
        (scene.map_kind, mv, 5, "map_kind"),
    ):
        if value.dtype != torch.long or ((value[mask] < 0) | (value[mask] >= upper)).any():
            raise ValueError(f"{name} is outside its categorical vocabulary")
    return replace(
        scene, agents=torch.where(av[..., None], scene.agents, 0),
        agent_class=torch.where(av.any(1), scene.agent_class, 0),
        confidence=torch.where(cv, scene.confidence, 0), confidence_valid=cv,
        source=torch.where(av[..., None], scene.source, 0),
        lineage=torch.where(av, scene.lineage, 0),
        ambiguity=torch.where(av, scene.ambiguity, 0),
        map_features=torch.where(mv[..., None], scene.map_features, 0),
        map_kind=torch.where(mv, scene.map_kind, 0),
        map_source=torch.where(mv[..., None], scene.map_source, 0),
        candidates=torch.where(scene.candidate_valid[..., None, None], executed, 0),
        rule_margins=torch.where(rv, margins, 0), rule_valid=rv,
    )


class RevisionV5Model(nn.Module):
    """No labels in forward; independent parameters for utility and event risk.

    The default frozen factual anchor is *not* a trained checkpoint. Callers must
    load and independently qualify an anchor before any substantive experiment.
    ``no_action`` disables only the actor response: the scorer still sees ego
    execution geometry. The learned unmatched-actor token permits risk for
    unobserved/new actors; it does not invent their future coordinates.
    Padding is exactly masked, but valid actors are still encoded and pooled:
    no certified relevance mask currently guarantees invariance to adding an
    observed, physically irrelevant actor. Padding tests do not establish that
    stronger intervention property.

    In ``current_only`` mode, the frozen anchor encoder is shared for a matched
    history representation, while its rollout and all response modules are
    skipped. World-shaped outputs are explicitly marked unavailable predictions:
    states repeat current observations, existence is an observation-valid mask,
    and variance is a fixed zero placeholder. They must not be used as scientific
    world/response predictions or as a successful response training experiment.
    Utility already uses an independent current/history encoder in both modes.
    """

    def __init__(self, config: RevisionV5Config | None = None,
                 anchor: CoordiWorld | None = None):
        super().__init__()
        self.config = config or RevisionV5Config()
        c, d = self.config, self.config.hidden_dim
        if c.dt <= 0 or d % c.heads or c.rule_temperature <= 0:
            raise ValueError("dt, rule temperature and attention dimensions are invalid")
        if c.risk_world_mode not in {"predicted", "current_only"}:
            raise ValueError("risk_world_mode must be predicted or current_only")
        if c.risk_world_mode == "current_only" and not c.freeze_anchor:
            raise ValueError("current_only requires the matched history encoder to remain frozen")
        self.anchor = anchor if anchor is not None else CoordiWorld(c.legacy_config())
        if self.anchor.config.action_conditioned:
            raise ValueError("the factual anchor must be action independent")
        for key in ("hidden_dim", "dt", "use_provenance", "use_confidence"):
            if getattr(self.anchor.config, key) != getattr(c, key):
                raise ValueError(f"anchor config does not match {key}")
        if c.freeze_anchor:
            self.anchor.requires_grad_(False)
            self.anchor.eval()
        self.response_action = mlp(9, d, d)
        self.response_temporal = nn.GRU(d, d, batch_first=True)
        self.response_head = mlp(d * 2 + 8, d, 3)
        nn.init.zeros_(self.response_head[-1].weight)
        nn.init.zeros_(self.response_head[-1].bias)
        self.risk_actor = mlp(d + 17, d, d)
        self.risk_ego = mlp(9, d, d)
        self.risk_query = mlp(d * 2, d, d)
        self.unmatched_actor = nn.Parameter(torch.zeros(1, 1, 1, 1, d))
        self.risk_temporal = nn.GRU(d * 3, d, batch_first=True)
        self.risk_head = nn.Linear(d, 2)
        with torch.no_grad():
            self.risk_head.bias.copy_(torch.tensor([-4.0, -4.7]))
        # All utility representation parameters are distinct from world/risk.
        self.value_encoder = StructuredHistoryEncoder(c.legacy_config())
        self.value_action = mlp(9, d, d)
        self.value_temporal = nn.GRU(d, d, batch_first=True)
        self.value_head = mlp(d * 3, d, 2)
        self.register_buffer("rule_temperatures", torch.full((4,), c.rule_temperature))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_anchor:
            self.anchor.eval()
        return self

    def _action_features(self, x: SceneInputs) -> Tensor:
        action = x.candidates
        b, m, h, _ = action.shape
        velocity = torch.diff(torch.cat((x.ego[:, -1:, :2][:, None].expand(-1, m, -1, -1),
                                        action[..., :2]), dim=2), dim=2) / self.config.dt
        time = (torch.arange(1, h + 1, device=action.device, dtype=action.dtype)
                * self.config.dt / 4.0).view(1, 1, h, 1).expand(b, m, -1, -1)
        dimensions = x.ego[:, -1, 5:7][:, None, None].expand(-1, m, h, -1) / 10
        return torch.cat((action[..., :2] / 50, action[..., 2:3].sin(),
                          action[..., 2:3].cos(), velocity / 20, dimensions, time), -1)

    def _world(self, x: SceneInputs, reference: Tensor, action_features: Tensor):
        b, m, h, _ = x.candidates.shape
        n = x.agents.shape[2]
        batch = torch.arange(b, device=x.ego.device)
        factual_x = x.select_candidates(reference)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.config.freeze_anchor):
            actor_context, map_context = self.anchor.encode(factual_x)
            if n:
                anchor = self.anchor.rollout(factual_x, (actor_context, map_context))
                anchor = {name: value[:, 0] for name, value in anchor.items()}
            else:
                anchor = {
                    "states": x.agents.new_zeros((b, h, 0, 7)),
                    "existence": x.agents.new_zeros((b, h, 0)),
                    "variance": x.agents.new_zeros((b, h, 0, 2)),
                }
        current_valid = x.agent_valid[:, -1]
        anchor["states"] = anchor["states"] * current_valid[:, None, :, None]
        anchor["existence"] = anchor["existence"] * current_valid[:, None]
        anchor["variance"] = anchor["variance"] * current_valid[:, None, :, None]
        expanded = anchor["states"][:, None].expand(-1, m, -1, -1, -1)
        if self.config.action_conditioned and n:
            action, _ = self.response_temporal(self.response_action(action_features).reshape(b * m, h, -1))
            action = action.reshape(b, m, h, 1, -1).expand(-1, -1, -1, n, -1)
            context = actor_context[:, None, None].expand(-1, m, h, -1, -1)
            raw = torch.tanh(self.response_head(torch.cat((context, action, normalized_state(expanded)), -1)))
            # The same deterministic computation is subtracted, giving exactly
            # zero response for a0, including during model.train().
            centered = raw - raw[batch, reference][:, None]
            position_delta = centered[..., :2] * self.config.response_position_scale
            yaw_delta = centered[..., 2:3] * self.config.response_yaw_scale
            velocity_delta = torch.diff(torch.cat((torch.zeros_like(position_delta[:, :, :1]),
                                                   position_delta), 2), dim=2) / self.config.dt
            delta = torch.cat((position_delta, yaw_delta, velocity_delta,
                               torch.zeros_like(expanded[..., 5:7])), -1)
        else:
            delta = torch.zeros_like(expanded)
        mask = x.candidate_valid[:, :, None, None] & current_valid[:, None, None]
        delta = delta * mask[..., None]
        states = expanded + delta
        states = torch.cat((states[..., :2], wrap_angle(states[..., 2:3]), states[..., 3:]), -1)
        states = states * mask[..., None]
        return {
            "states": states, "response_delta": delta,
            "existence": anchor["existence"][:, None].expand(-1, m, -1, -1) * mask,
            "variance": anchor["variance"][:, None].expand(-1, m, -1, -1, -1) * mask[..., None],
            "anchor_states": anchor["states"],
        }, actor_context, map_context

    def _current_actor_information(self, x):
        """Static observation placeholders with no future-model dependency.

        Repeating position with nonzero measured velocity is intentional: these
        are features available now, not a physically integrated motion forecast.
        The fixed variance channels only preserve risk-head parameter dimensions.
        """
        b, m, h, _ = x.candidates.shape
        current = x.agents[:, -1]
        valid = x.agent_valid[:, -1]
        anchor = current[:, None].expand(-1, h, -1, -1)
        mask = x.candidate_valid[:, :, None, None] & valid[:, None, None]
        states = anchor[:, None].expand(-1, m, -1, -1, -1) * mask[..., None]
        return {
            "states": states,
            "response_delta": torch.zeros_like(states),
            "existence": mask.expand(b, m, h, -1).to(states.dtype),
            "variance": states.new_zeros((*states.shape[:-1], 2)),
            "anchor_states": anchor * valid[:, None, :, None],
        }

    def _current_world(self, x, reference):
        # Only the representation is shared with the qualified frozen anchor.
        # In particular, no anchor.rollout or response forward is executed.
        with torch.no_grad():
            actors, maps = self.anchor.encode(x.select_candidates(reference))
        return self._current_actor_information(x), actors, maps

    def _risk(self, x, world, actor_context, map_context, action_features):
        c = self.config
        if c.risk_world_mode == "current_only":
            # Do not trust even an externally supplied/perturbed world dictionary
            # in this mode: the risk path is a function of observations only.
            world = self._current_actor_information(x)
        if c.detach_world_for_risk:
            states, variance, existence = (world[name].detach() for name in ("states", "variance", "existence"))
            actor_context, map_context = actor_context.detach(), map_context.detach()
        else:
            states, variance, existence = (world[name] for name in ("states", "variance", "existence"))
        b, m, h, n, _ = states.shape
        d = c.hidden_dim
        ego_box = rear_axle_to_box(x.candidates, x.ego[:, -1, 5:7][:, None, None],
                                  x.ego_rear_axle_to_center[:, None, None])
        boxes = torch.cat((states[..., :3], states[..., 5:7]), -1)
        gap = sat_clearance(ego_box[..., None, :], boxes)
        delta = states[..., :2] - ego_box[..., None, :2]
        relative_yaw = states[..., 2] - ego_box[..., None, 2]
        ego_velocity = action_features[..., 4:6] * 20
        relative_velocity = states[..., 3:5] - ego_velocity[..., None, :]
        ego_dims = ego_box[..., None, 3:5].expand(-1, -1, -1, n, -1)
        time = action_features[..., -1:, None].transpose(-1, -2).expand(-1, -1, -1, n, -1)
        geometry = torch.cat((delta / 50, relative_yaw.sin()[..., None], relative_yaw.cos()[..., None],
                              relative_velocity / 20, gap[..., None] / 10,
                              variance.clamp_min(0.05).log() / 5, existence[..., None],
                              states[..., 5:7] / 10, ego_dims / 10,
                              ego_velocity[..., None, :].expand(-1, -1, -1, n, -1) / 20,
                              time), -1)
        actor = self.risk_actor(torch.cat((actor_context[:, None, None].expand(-1, m, h, -1, -1), geometry), -1))
        # This is learned pooling over actors, never an actor-wise noisy-OR.
        actor = torch.cat((actor, self.unmatched_actor.expand(b, m, h, 1, d)), -2)
        valid = torch.cat((x.agent_valid[:, -1], torch.ones((b, 1), device=x.ego.device, dtype=torch.bool)), -1)
        maps = _masked_mean(map_context, x.map_valid, 1)[:, None, None].expand(-1, m, h, -1)
        ego = self.risk_ego(action_features)
        query = self.risk_query(torch.cat((ego, maps), -1))
        weights = torch.softmax((actor * query[..., None, :]).sum(-1).div(d ** 0.5)
                                .masked_fill(~valid[:, None, None], -torch.inf), -1)
        pooled = (actor * weights[..., None]).sum(-2)
        temporal, _ = self.risk_temporal(torch.cat((pooled, ego, maps), -1).reshape(b * m, h, d * 3))
        raw_logits = self.risk_head(temporal).reshape(b, m, h, 2).float()
        raw_hazard = raw_logits.sigmoid()
        raw_log_survival = torch.nn.functional.logsigmoid(-raw_logits).cumsum(2)
        raw_cdf = -torch.expm1(raw_log_survival)
        # At-fault events are a subset of physical collisions. Project the two
        # learned CDFs into this logical constraint at every horizon. Min of two
        # nondecreasing CDFs is itself a CDF; differencing gives valid hazards.
        # Work in log survival throughout the projection. Subtracting rounded
        # FP32 CDFs would turn later hazards into zero once the CDF rounds to 1,
        # and destroy first-event / no-event likelihoods and their gradients.
        log_survival = torch.stack((raw_log_survival[..., 0],
                                    torch.maximum(raw_log_survival[..., 0], raw_log_survival[..., 1])), -1)
        previous_log_survival = torch.cat((torch.zeros_like(log_survival[:, :, :1]),
                                           log_survival[:, :, :-1]), 2)
        log_hazard_survival = (log_survival - previous_log_survival).clamp_max(0)
        cdf = -torch.expm1(log_survival)
        hazard = -torch.expm1(log_hazard_survival)
        mask = x.candidate_valid
        return {
            "physical_collision": cdf[:, :, -1, 0] * mask,
            "at_fault_collision": cdf[:, :, -1, 1] * mask,
            "hazard": hazard * mask[..., None, None],
            "hazard_logits": torch.logit(hazard.clamp(1e-7, 1 - 1e-7)) * mask[..., None, None],
            "raw_hazard_logits": raw_logits * mask[..., None, None],
            "raw_hazard": raw_hazard * mask[..., None, None],
            "raw_event_cdf": raw_cdf * mask[..., None, None],
            "event_cdf": cdf * mask[..., None, None],
            "event_log_survival": log_survival * mask[..., None, None],
            "hazard_log_survival": log_hazard_survival * mask[..., None, None],
            "risk_geometry": geometry * x.agent_valid[:, None, -1:, :, None],
            "actor_attention": weights * mask[..., None, None],
        }

    def _value(self, x, action_features):
        b, m, h, _ = action_features.shape
        actors, maps = self.value_encoder(x)
        actor = _masked_mean(actors, x.agent_valid[:, -1], 1)[:, None].expand(-1, m, -1)
        map_state = _masked_mean(maps, x.map_valid, 1)[:, None].expand(-1, m, -1)
        _, action = self.value_temporal(self.value_action(action_features).reshape(b * m, h, -1))
        action = action[-1].reshape(b, m, -1)
        values = self.value_head(torch.cat((actor, map_state, action), -1)).sigmoid()
        return {"utility": values[..., 0] * x.candidate_valid,
                "progress": values[..., 1] * x.candidate_valid}

    def forward(self, scene: SceneInputs, executed_candidates: Tensor,
                reference_action: Tensor | int,
                *, executed_rule_margins: Tensor | None = None,
                executed_rule_valid: Tensor | None = None) -> dict[str, Tensor]:
        x = _clean_scene(scene, executed_candidates, executed_rule_margins, executed_rule_valid)
        # Separate provenance factors without altering observation content or
        # confidence. use_provenance remains the legacy aggregate switch.
        if not self.config.use_source:
            x = replace(x, source=torch.zeros_like(x.source), map_source=torch.zeros_like(x.map_source))
        if not self.config.use_lineage:
            x = replace(x, lineage=torch.zeros_like(x.lineage))
        if not self.config.use_ambiguity:
            x = replace(x, ambiguity=torch.zeros_like(x.ambiguity))
        if x.agents.shape[2] > self.config.max_agents or x.ego.shape[1] > self.config.max_history:
            raise ValueError("scene exceeds configured actor/history capacity")
        reference = _reference_indices(reference_action, x.candidate_valid)
        action_features = self._action_features(x)
        if self.config.risk_world_mode == "current_only":
            world, actors, maps = self._current_world(x, reference)
        else:
            world, actors, maps = self._world(x, reference, action_features)
        risk = self._risk(x, world, actors, maps, action_features)
        violation, available = map_risk(x, self.rule_temperatures, self.config.violation_beta)
        return {**world, **risk, **self._value(x, action_features),
                "violation": violation * x.candidate_valid,
                "violation_available": available & x.candidate_valid,
                "candidate_valid": x.candidate_valid, "reference_action": reference,
                "actor_valid": x.agent_valid[:, -1],
                "world_prediction_available": torch.tensor(
                    self.config.risk_world_mode == "predicted", device=x.ego.device, dtype=torch.bool)}
