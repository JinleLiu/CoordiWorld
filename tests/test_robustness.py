"""Synthetic tests for robustness perturbations."""

from __future__ import annotations

from coordiworld.evaluation.robustness import (
    apply_confidence_noise,
    apply_evidence_dropout,
    compute_ranking_stability,
    mask_provenance_channels,
)


def make_agent_tensor() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.5, 1.0, 0.1, 0.0, 0.1, 1.0, 2.0, 1.0, 0.3],
        [4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.6, 1.0, 0.1, 0.0, 0.1, 1.0, 3.0, 2.0, 0.5],
    ]


def test_confidence_noise_is_deterministic_and_clamped() -> None:
    first = apply_confidence_noise(make_agent_tensor(), [1, 1], noise_std=0.2, seed=4)
    second = apply_confidence_noise(make_agent_tensor(), [1, 1], noise_std=0.2, seed=4)

    assert first == second
    assert all(0.0 <= token[7] <= 1.0 for token in first)
    assert first[0][0] == 1.0


def test_provenance_masking_preserves_geometry() -> None:
    masked = mask_provenance_channels(make_agent_tensor(), [1, 1])

    assert masked[0][:2] == [1.0, 0.0]
    assert masked[0][13] == 0.0
    assert masked[0][14] == 0.0
    assert masked[1][13] == 0.0


def test_evidence_dropout_masks_entities() -> None:
    dropped = apply_evidence_dropout(make_agent_tensor(), [1, 1], dropout_prob=1.0, seed=0)

    assert dropped.dropped_indices == [0, 1]
    assert dropped.agent_mask == [0, 0]
    assert dropped.agent_tensor[0] == [0.0] * len(make_agent_tensor()[0])


def test_ranking_stability_reports_top1_and_kendall() -> None:
    stable = compute_ranking_stability([0.1, 0.3, 0.5], [0.12, 0.31, 0.49])
    unstable = compute_ranking_stability([0.1, 0.3, 0.5], [0.6, 0.3, 0.1])

    assert stable.top1_unchanged is True
    assert stable.kendall == 1.0
    assert stable.mean_abs_score_delta > 0.0
    assert unstable.top1_unchanged is False
    assert unstable.kendall < 0.0
