"""Synthetic tests for CoordiWorld risk heads and calibrated scoring."""

from __future__ import annotations

from dataclasses import fields

from coordiworld.risks.calibration import fit_calibrator, load_calibrator, save_calibrator
from coordiworld.risks.scoring import CandidateRiskScore, ScoreWeights, score_candidates
from coordiworld.scene_summary.schema import MapToken


def make_drivable_map() -> list[MapToken]:
    return [
        MapToken(
            id="drive-0",
            type="drivable_area",
            polyline=None,
            polygon=[[-5.0, -3.0], [20.0, -3.0], [20.0, 3.0], [-5.0, 3.0]],
            traffic_state=None,
            rule_attributes={},
        ),
        MapToken(
            id="lane-0",
            type="lane_centerline",
            polyline=[[-5.0, 0.0], [20.0, 0.0]],
            polygon=None,
            traffic_state=None,
            rule_attributes={},
        ),
        MapToken(
            id="stop-0",
            type="stop_line",
            polyline=[[8.0, -2.0], [8.0, 2.0]],
            polygon=None,
            traffic_state=None,
            rule_attributes={"requires_stop": True},
        ),
        MapToken(
            id="light-0",
            type="traffic_light",
            polyline=[[12.0, -2.0], [12.0, 2.0]],
            polygon=None,
            traffic_state="red",
            rule_attributes={},
        ),
    ]


def make_covariance(var: float, *, agents: int = 1, horizon: int = 4) -> list[list[list[float]]]:
    return [[[var, var] for _ in range(horizon)] for _ in range(agents)]


def test_collision_candidate_j_higher_than_safe_candidate() -> None:
    safe_candidate = [[0.0, 4.5, 0.0], [2.0, 4.5, 0.0], [4.0, 4.5, 0.0], [6.0, 4.5, 0.0]]
    collision_candidate = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    predicted_agents = [[[4.0, 0.0, 0.0] for _ in range(4)]]

    scores = score_candidates(
        candidate_trajectories=[safe_candidate, collision_candidate],
        predicted_agent_trajectories=predicted_agents,
        map_tokens=[],
        covariance=make_covariance(0.1),
        weights=ScoreWeights(lambda_c=1.0, lambda_v=0.0, lambda_u=0.0),
    )

    assert scores[1].collision.p_collision > scores[0].collision.p_collision
    assert scores[1].j_score > scores[0].j_score


def test_violation_candidate_j_higher_than_compliant_candidate() -> None:
    compliant = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    violation = [[0.0, 5.0, 0.0], [2.0, 5.0, 0.0], [4.0, 5.0, 0.0], [6.0, 5.0, 0.0]]
    far_agent = [[[30.0, 0.0, 0.0] for _ in range(4)]]

    scores = score_candidates(
        candidate_trajectories=[compliant, violation],
        predicted_agent_trajectories=far_agent,
        map_tokens=make_drivable_map(),
        covariance=make_covariance(0.1),
        weights=ScoreWeights(lambda_c=0.0, lambda_v=1.0, lambda_u=0.0),
    )

    assert scores[1].violation.components["drivable_area"] > 0.8
    assert scores[1].j_score > scores[0].j_score


def test_uncertainty_contributes_to_j_score() -> None:
    candidate = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    agent = [[[30.0, 0.0, 0.0] for _ in range(4)]]

    low, high = score_candidates(
        candidate_trajectories=[candidate, candidate],
        predicted_agent_trajectories=agent,
        map_tokens=[],
        covariance=[make_covariance(0.1), make_covariance(2.0)],
        weights=ScoreWeights(lambda_c=0.0, lambda_v=0.0, lambda_u=1.0),
        u95=4.0,
    )

    assert high.uncertainty.u_bar > low.uncertainty.u_bar
    assert high.j_score > low.j_score


def test_calibration_save_load_roundtrip(tmp_path) -> None:
    calibrator = fit_calibrator(
        scores=[0.05, 0.2, 0.6, 0.9],
        labels=[0, 0, 1, 1],
        n_bins=4,
        method="isotonic",
    )
    path = tmp_path / "collision_calibrator.json"

    save_calibrator(calibrator, path)
    loaded = load_calibrator(path)

    assert loaded == calibrator
    assert loaded.apply(0.1) <= loaded.apply(0.8)
    assert loaded.apply_many([0.1, 0.8]) == [loaded.apply(0.1), loaded.apply(0.8)]


def test_candidate_risk_score_field_contract_is_stable() -> None:
    assert [field.name for field in fields(CandidateRiskScore)] == [
        "candidate_index",
        "collision",
        "violation",
        "uncertainty",
        "p_coll_calibrated",
        "p_viol_calibrated",
        "j_score",
    ]
