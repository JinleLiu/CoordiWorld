"""Final calibrated CoordiWorld risk score aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from coordiworld.risks.calibration import BinningCalibrator
from coordiworld.risks.collision import CollisionRiskResult, compute_collision_risk
from coordiworld.risks.rule_violation import RuleViolationResult, compute_rule_violation_risk
from coordiworld.risks.uncertainty import (
    PredictiveUncertaintyResult,
    compute_predictive_uncertainty,
)
from coordiworld.scene_summary.schema import MapToken

Trajectory = Sequence[Sequence[float]]


@dataclass(frozen=True)
class ScoreWeights:
    lambda_c: float = 1.0
    lambda_v: float = 1.0
    lambda_u: float = 0.25


@dataclass(frozen=True)
class CandidateRiskScore:
    candidate_index: int
    collision: CollisionRiskResult
    violation: RuleViolationResult
    uncertainty: PredictiveUncertaintyResult
    p_coll_calibrated: float
    p_viol_calibrated: float
    j_score: float


def compute_candidate_score(
    *,
    candidate_index: int,
    ego_trajectory: Trajectory,
    predicted_agent_trajectories: Sequence[Trajectory],
    map_tokens: Sequence[MapToken],
    covariance: Sequence[Sequence[Sequence[float] | Sequence[Sequence[float]]]],
    existence_probabilities: Sequence[Sequence[float]] | None = None,
    collision_calibrator: BinningCalibrator | None = None,
    violation_calibrator: BinningCalibrator | None = None,
    weights: ScoreWeights | None = None,
    u95: float = 4.0,
) -> CandidateRiskScore:
    """Compute lower-is-better calibrated score J for one candidate."""
    resolved_weights = weights or ScoreWeights()
    collision = compute_collision_risk(
        ego_trajectory,
        predicted_agent_trajectories,
        existence_probabilities=existence_probabilities,
    )
    violation = compute_rule_violation_risk(ego_trajectory, map_tokens)
    uncertainty = compute_predictive_uncertainty(
        covariance,
        existence_probabilities=existence_probabilities,
        u95=u95,
    )
    p_coll = _apply_calibrator(collision_calibrator, collision.p_collision)
    p_viol = _apply_calibrator(violation_calibrator, violation.p_violation)
    j_score = (
        resolved_weights.lambda_c * p_coll
        + resolved_weights.lambda_v * p_viol
        + resolved_weights.lambda_u * uncertainty.u_bar
    )
    return CandidateRiskScore(
        candidate_index=candidate_index,
        collision=collision,
        violation=violation,
        uncertainty=uncertainty,
        p_coll_calibrated=p_coll,
        p_viol_calibrated=p_viol,
        j_score=j_score,
    )


def score_candidates(
    *,
    candidate_trajectories: Sequence[Trajectory],
    predicted_agent_trajectories: Sequence[Sequence[Trajectory]] | Sequence[Trajectory],
    map_tokens: Sequence[MapToken],
    covariance: Sequence[Sequence[Sequence[Sequence[float] | Sequence[Sequence[float]]]]]
    | Sequence[Sequence[Sequence[float] | Sequence[Sequence[float]]]],
    existence_probabilities: (
        Sequence[Sequence[Sequence[float]]] | Sequence[Sequence[float]] | None
    ) = None,
    collision_calibrator: BinningCalibrator | None = None,
    violation_calibrator: BinningCalibrator | None = None,
    weights: ScoreWeights | None = None,
    u95: float = 4.0,
) -> list[CandidateRiskScore]:
    """Score a shared candidate pool. Lower J is better."""
    candidate_count = len(candidate_trajectories)
    return [
        compute_candidate_score(
            candidate_index=index,
            ego_trajectory=candidate,
            predicted_agent_trajectories=_candidate_slice(
                predicted_agent_trajectories,
                index,
                candidate_count,
            ),
            map_tokens=map_tokens,
            covariance=_candidate_slice(covariance, index, candidate_count),
            existence_probabilities=_candidate_slice_optional(
                existence_probabilities,
                index,
                candidate_count,
            ),
            collision_calibrator=collision_calibrator,
            violation_calibrator=violation_calibrator,
            weights=weights,
            u95=u95,
        )
        for index, candidate in enumerate(candidate_trajectories)
    ]


def _apply_calibrator(calibrator: BinningCalibrator | None, value: float) -> float:
    return value if calibrator is None else calibrator.apply(value)


def _candidate_slice(values: Sequence[object], index: int, candidate_count: int) -> object:
    if len(values) == candidate_count:
        return values[index]
    return values


def _candidate_slice_optional(
    values: Sequence[object] | None,
    index: int,
    candidate_count: int,
) -> object | None:
    if values is None:
        return None
    return _candidate_slice(values, index, candidate_count)
