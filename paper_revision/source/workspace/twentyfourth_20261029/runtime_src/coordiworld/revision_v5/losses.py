"""Label-only objectives for V5; missing tasks never become negative labels."""
from dataclasses import dataclass, fields, replace
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from ..geometry import wrap_angle


@dataclass
class RevisionV5Targets:
    # Paired simulator rollouts, aligned by actor identity, same domain, initial
    # state and exogenous RNG. Not independent logged futures per candidate.
    paired_states: Tensor                 # [B,M,H,N,7]
    paired_valid: Tensor                  # [B,M,H,N]
    target_valid: Tensor                  # [B,M], generic availability
    domain_id: str
    paired_domain_id: str
    paired_initial_state_ids: tuple        # B rows of M equal strings
    paired_rng_ids: tuple                  # B rows of M equal strings
    physical_collision: Tensor | None = None  # [B,M], binary
    at_fault_collision: Tensor | None = None  # [B,M], binary
    utility: Tensor | None = None          # [B,M], normalized [0,1]
    progress: Tensor | None = None         # [B,M], normalized [0,1]
    physical_valid: Tensor | None = None
    at_fault_valid: Tensor | None = None
    utility_valid: Tensor | None = None
    progress_valid: Tensor | None = None
    physical_first_event: Tensor | None = None  # [B,M], -2 unknown, -1 none, 0..H-1
    at_fault_first_event: Tensor | None = None
    response_supervision_valid: Tensor | None = None  # [B,M]
    paired_existence: Tensor | None = None      # [B,M,H,N], explicit binary existence
    paired_existence_valid: Tensor | None = None  # availability, not inferred from visibility

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device)
                               for f in fields(self) if isinstance(getattr(self, f.name), Tensor)})


@dataclass
class RevisionV5LossConfig:
    response_weight: float = 1.0
    factual_weight: float = 0.1
    existence_weight: float = 0.1
    risk_weight: float = 1.0
    hazard_weight: float = 1.0
    utility_weight: float = 1.0
    progress_weight: float = 1.0
    consistency_weight: float = 0.1
    pairwise_weight: float = 0.1
    pairwise_min_utility_gap: float = 0.02
    pairwise_temperature: float = 0.1


def _mean(values: Tensor, mask: Tensor) -> Tensor:
    # where, not multiplication: unavailable label values may be NaN.
    return torch.where(mask, values, torch.zeros_like(values)).sum() / mask.sum().clamp_min(1)


def _bernoulli_nll(probability: Tensor, labels: Tensor) -> Tensor:
    # Explicit FP32 logs also work inside CUDA autocast, where probability-space
    # binary_cross_entropy is disallowed. Event probabilities come from a CDF,
    # so there is not a single unconstrained endpoint logit to use instead.
    probability = probability.float().clamp(1e-7, 1 - 1e-7)
    labels = labels.float()
    return -(labels * probability.log() + (1 - labels) * torch.log1p(-probability))


def _event_log_probability(log_survival: Tensor) -> Tensor:
    """Stable log(1 - survival), including almost certain event predictions.

    A projected interval can have exactly zero event probability. Use only the
    dtype's smallest normal positive value there so the loss remains finite;
    positive event probabilities are not clipped at an arbitrary 1e-7 floor.
    """
    log_survival = log_survival.float()
    safe = log_survival.clamp_max(-torch.finfo(log_survival.dtype).tiny)
    return (-torch.expm1(safe)).log()


def _event_nll(log_survival: Tensor, labels: Tensor) -> Tensor:
    # Endpoint collision labels are binary, so use where rather than 0 * log(0).
    return -torch.where(labels.bool(), _event_log_probability(log_survival), log_survival)


def _utility_pairwise(prediction: Tensor, labels: Tensor | None,
                      valid: Tensor, config: RevisionV5LossConfig):
    """One orientation per within-scene pair; all parameters belong to value.

    Both labels must be available, and their absolute gap must exceed the
    frozen practical-tie threshold. The threshold/temperature are protocol
    choices, not fitted on the final confirmation set. No risk/world signal is
    used to manufacture a utility preference.
    """
    zero = prediction.sum() * 0
    if labels is None:
        return zero, valid.new_zeros((), dtype=torch.long)
    m = prediction.shape[1]
    label_gap = labels[:, :, None] - labels[:, None, :]
    upper = torch.ones((m, m), device=valid.device, dtype=torch.bool).triu(diagonal=1)
    pairs = (valid[:, :, None] & valid[:, None, :] & upper[None]
             & (label_gap.abs() > config.pairwise_min_utility_gap))
    predicted_gap = prediction[:, :, None] - prediction[:, None, :]
    signed_gap = label_gap.sign() * predicted_gap / config.pairwise_temperature
    return _mean(F.softplus(-signed_gap), pairs), pairs.sum()


def _mask(targets, field, base):
    value = getattr(targets, field)
    if value is None:
        return base
    if value.shape != base.shape or value.dtype != torch.bool:
        raise ValueError(f"{field} must be a bool [B,M] tensor")
    return base & value


def _validate_pair_metadata(targets, mask):
    if not isinstance(targets.domain_id, str) or not targets.domain_id:
        raise ValueError("a named simulator domain is required")
    if targets.paired_domain_id != targets.domain_id:
        raise ValueError("response supervision must come from the same simulator domain")
    b, m = mask.shape
    for name in ("paired_initial_state_ids", "paired_rng_ids"):
        groups = getattr(targets, name)
        if len(groups) != b or any(len(row) != m for row in groups):
            raise ValueError(f"{name} must contain B rows of M identifiers")
        for i, row in enumerate(groups):
            used = [row[j] for j in range(m) if bool(mask[i, j])]
            if any(not isinstance(item, str) or not item for item in used) or len(set(used)) > 1:
                raise ValueError(f"{name} must match within each paired scene")


def _task_target(targets, name, mask, *, binary=False):
    labels = getattr(targets, name)
    if labels is None:
        return None
    if labels.shape != mask.shape:
        raise ValueError(f"{name} must have shape [B,M]")
    selected = labels[mask]
    if not torch.isfinite(selected).all() or ((selected < 0) | (selected > 1)).any():
        raise ValueError(f"available {name} labels must be finite and in [0,1]")
    if binary and ((selected != 0) & (selected != 1)).any():
        raise ValueError(f"{name} labels must be binary")
    return torch.where(mask, labels, torch.zeros_like(labels))


def _first_event_loss(hazard, times, labels, mask, log_hazard_survival=None):
    if times is None:
        return hazard.sum() * 0
    b, m, h = hazard.shape
    if times.shape != (b, m) or times.dtype != torch.long:
        raise ValueError("first-event labels must be long [B,M] tensors")
    known = mask & (times != -2)
    if ((times[known] < -1) | (times[known] >= h)).any():
        raise ValueError("first event must be -2, -1, or a zero-based horizon index")
    if labels is not None and not torch.equal(times[known].ge(0), labels[known].bool()):
        raise ValueError("first-event time contradicts its collision endpoint label")
    time = torch.arange(h, device=hazard.device).view(1, 1, h)
    survived = known[..., None] & ((times[..., None] == -1) | (time < times[..., None]))
    occurred = known[..., None] & (time == times[..., None])
    if log_hazard_survival is None:
        # Compatibility for externally constructed output dictionaries. Native
        # V5 outputs always carry the stable log-survival representation.
        log_hazard_survival = torch.log1p(-hazard.float().clamp(1e-7, 1 - 1e-7))
    per_candidate = -(torch.where(survived, log_hazard_survival, 0).sum(-1)
                      + torch.where(occurred, _event_log_probability(log_hazard_survival), 0).sum(-1))
    return _mean(per_candidate, known)


def revision_v5_loss(output: dict[str, Tensor], targets: RevisionV5Targets,
                     config: RevisionV5LossConfig | None = None) -> dict[str, Tensor]:
    """Return separate stage objectives and their weighted sum.

    A response stage uses ``world``; an event stage uses ``risk_total``; a value
    stage uses ``value_total``. Default model detach boundaries prevent value or
    event losses from modifying the world. Simultaneous ``total`` is available
    for synthetic integration checks, not a mandate to mix research stages.
    Missing event times permit endpoint training but do not certify temporal
    calibration. Metadata checks enforce declared pairing, not its authenticity.
    """
    c = config or RevisionV5LossConfig()
    for field in fields(c):
        value = getattr(c, field.name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field.name} must be finite and nonnegative")
    if c.pairwise_temperature == 0:
        raise ValueError("pairwise_temperature must be positive")
    prediction = output["states"]
    b, m, h, n, channels = prediction.shape
    if targets.target_valid.shape != (b, m) or targets.target_valid.dtype != torch.bool:
        raise ValueError("target_valid must be bool [B,M]")
    base = output["candidate_valid"] & targets.target_valid
    response_candidates = _mask(targets, "response_supervision_valid", base)
    _validate_pair_metadata(targets, response_candidates)
    if targets.paired_states.shape != prediction.shape or targets.paired_valid.shape != (b, m, h, n):
        raise ValueError("paired states/valid must align with model actors and candidate horizons")
    if targets.paired_valid.dtype != torch.bool or channels != 7:
        raise ValueError("paired validity must be bool and states must have seven channels")
    current_actor = output["actor_valid"][:, None, None]
    valid = targets.paired_valid & response_candidates[..., None, None] & current_actor
    if not torch.isfinite(targets.paired_states[valid]).all():
        raise ValueError("available paired states must be finite")
    truth = torch.where(valid[..., None], targets.paired_states, 0)
    reference = output["reference_action"]
    batch = torch.arange(b, device=prediction.device)
    ref_truth = truth[batch, reference]
    ref_valid = valid[batch, reference]
    pair_valid = valid & ref_valid[:, None]
    nonreference = torch.arange(m, device=prediction.device)[None] != reference[:, None]
    response_valid = pair_valid & nonreference[..., None, None]
    target_delta = truth - ref_truth[:, None]
    target_delta = torch.cat((target_delta[..., :2], wrap_angle(target_delta[..., 2:3]),
                              target_delta[..., 3:5]), -1)
    error = output["response_delta"][..., :5] - target_delta
    error = torch.cat((error[..., :2], wrap_angle(error[..., 2:3]), error[..., 3:]), -1)
    scale = error.new_tensor([1, 1, 1, 2, 2])
    response = _mean(F.smooth_l1_loss(error / scale, torch.zeros_like(error), reduction="none").mean(-1), response_valid)
    factual_error = output["anchor_states"][..., :2] - ref_truth[..., :2]
    factual = _mean(F.smooth_l1_loss(factual_error, torch.zeros_like(factual_error), reduction="none").mean(-1), ref_valid)
    # Missing future states are not evidence of physical non-existence.
    existence = output["existence"].sum() * 0
    exists_mask = torch.zeros_like(targets.paired_valid)
    if (targets.paired_existence is None) != (targets.paired_existence_valid is None):
        raise ValueError("provide paired_existence and paired_existence_valid together")
    if targets.paired_existence is not None:
        labels, label_valid = targets.paired_existence, targets.paired_existence_valid
        if labels.shape != valid.shape or label_valid.shape != valid.shape or label_valid.dtype != torch.bool:
            raise ValueError("paired existence labels/valid must align with [B,M,H,N]")
        exists_mask = response_candidates[..., None, None] & current_actor & label_valid
        selected = labels[exists_mask]
        if not torch.isfinite(selected).all() or ((selected != 0) & (selected != 1)).any():
            raise ValueError("available paired_existence labels must be binary")
        labels = torch.where(exists_mask, labels, 0)
        existence = _mean(_bernoulli_nll(output["existence"], labels), exists_mask)
    risk_zero = output["hazard"].sum() * 0
    value_zero = output["utility"].sum() * 0
    risk, hazard = risk_zero, risk_zero
    clean_labels = {}
    event_supervised_elements = base.new_zeros((), dtype=torch.long)
    for name, mask_name, time_name, index in (
        ("physical_collision", "physical_valid", "physical_first_event", 0),
        ("at_fault_collision", "at_fault_valid", "at_fault_first_event", 1),
    ):
        mask = _mask(targets, mask_name, base)
        labels = _task_target(targets, name, mask, binary=True)
        clean_labels[name] = (labels, mask)
        if labels is not None:
            endpoint_nll = (_event_nll(output["event_log_survival"][:, :, -1, index], labels)
                            if "event_log_survival" in output else _bernoulli_nll(output[name], labels))
            risk = risk + _mean(endpoint_nll, mask)
        log_hazard_survival = (output["hazard_log_survival"][..., index]
                               if "hazard_log_survival" in output else None)
        hazard = hazard + _first_event_loss(output["hazard"][..., index],
                                            getattr(targets, time_name), labels, mask, log_hazard_survival)
        times = getattr(targets, time_name)
        supervised = mask if labels is not None else torch.zeros_like(mask)
        if times is not None:
            supervised = supervised | (mask & (times != -2))
        event_supervised_elements = event_supervised_elements + supervised.sum()
    physical, pm = clean_labels["physical_collision"]
    fault, fm = clean_labels["at_fault_collision"]
    if physical is not None and fault is not None and ((fault > physical) & pm & fm).any():
        raise ValueError("at-fault collision cannot be positive without physical collision")
    physical_time, fault_time = targets.physical_first_event, targets.at_fault_first_event
    if physical_time is not None and fault is not None:
        if ((physical_time == -1) & (fault == 1) & pm & fm).any():
            raise ValueError("at-fault collision contradicts known absence of physical collision")
    if fault_time is not None and physical is not None:
        if ((fault_time >= 0) & (physical == 0) & pm & fm).any():
            raise ValueError("at-fault first event contradicts physical collision endpoint")
    if physical_time is not None and fault_time is not None:
        known = pm & fm & (physical_time != -2) & (fault_time >= 0)
        if (known & ((physical_time == -1) | (physical_time > fault_time))).any():
            raise ValueError("physical first event must precede or equal at-fault first event")
    task_losses = {}
    task_counts = {}
    pairwise, pairwise_supervised_pairs = value_zero, base.new_zeros((), dtype=torch.long)
    for name in ("utility", "progress"):
        mask = _mask(targets, name + "_valid", base)
        labels = _task_target(targets, name, mask)
        task_counts[name + "_supervised_elements"] = mask.sum() if labels is not None else mask.new_zeros((), dtype=torch.long)
        task_losses[name] = value_zero if labels is None else _mean(
            F.smooth_l1_loss(output[name], labels, reduction="none"), mask)
        if name == "utility":
            pairwise, pairwise_supervised_pairs = _utility_pairwise(output[name], labels, mask, c)
    raw_cdf = output["raw_event_cdf"]
    consistency = _mean(F.relu(raw_cdf[..., 1] - raw_cdf[..., 0]).square().mean(-1), base)
    world = c.response_weight * response + c.factual_weight * factual + c.existence_weight * existence
    risk_total = c.risk_weight * risk + c.hazard_weight * hazard + c.consistency_weight * consistency
    value_total = (c.utility_weight * task_losses["utility"]
                   + c.progress_weight * task_losses["progress"]
                   + c.pairwise_weight * pairwise)
    return {"total": world + risk_total + value_total, "world": world,
            "response": response, "factual": factual, "existence": existence,
            "risk": risk, "hazard": hazard, "consistency": consistency,
            "risk_total": risk_total, "value_total": value_total,
            "response_supervised_elements": response_valid.sum(),
            "factual_supervised_elements": ref_valid.sum(),
            "existence_supervised_elements": exists_mask.sum(),
            "event_supervised_elements": event_supervised_elements,
            "value_supervised_elements": sum(task_counts.values()),
            "pairwise": pairwise,
            "pairwise_supervised_pairs": pairwise_supervised_pairs,
            **task_counts, **task_losses}
