"""Synthetic tests for calibration metrics and NAVSIM metric stubs."""

from __future__ import annotations

import pytest

from coordiworld.evaluation.calibration_metrics import (
    brier_score,
    compute_calibration_metrics,
    expected_calibration_error,
    reliability_bins,
)
from coordiworld.evaluation.navsim_metrics import (
    NAVSIMMetricAdapter,
    NAVSIMMetricUnavailableError,
)


def test_brier_score_matches_manual_value() -> None:
    probabilities = [0.1, 0.8, 0.4, 0.9]
    labels = [0, 1, 0, 1]

    assert brier_score(probabilities, labels) == pytest.approx(
        ((0.1 - 0) ** 2 + (0.8 - 1) ** 2 + (0.4 - 0) ** 2 + (0.9 - 1) ** 2) / 4
    )


def test_ece_matches_fixed_width_bins() -> None:
    probabilities = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]

    assert expected_calibration_error(probabilities, labels, n_bins=2) == pytest.approx(0.15)


def test_reliability_bins_report_counts_and_gaps() -> None:
    bins = reliability_bins([0.1, 0.4, 0.7, 0.9], [0, 0, 1, 1], n_bins=2)

    assert len(bins) == 2
    assert bins[0].count == 2
    assert bins[1].count == 2
    assert bins[0].confidence == pytest.approx(0.25)
    assert bins[1].accuracy == pytest.approx(1.0)


def test_compute_calibration_metrics_bundle() -> None:
    metrics = compute_calibration_metrics([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], n_bins=2)

    assert metrics.ece == pytest.approx(0.15)
    assert metrics.brier_score == pytest.approx(0.025)
    assert len(metrics.bins) == 2


def test_navsim_adapter_raises_clear_error_without_wrapper() -> None:
    with pytest.raises(NAVSIMMetricUnavailableError, match="will not fabricate EPDMS"):
        NAVSIMMetricAdapter().evaluate([])


def test_navsim_adapter_dry_run_does_not_generate_real_epdms() -> None:
    result = NAVSIMMetricAdapter(dry_run=True).evaluate([{"synthetic": True}])

    assert result.dry_run is True
    assert result.metrics["EPDMS"] is None
    assert "no real EPDMS" in result.message
