"""NAVSIM metric adapter interface and dry-run stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class NAVSIMMetricUnavailableError(RuntimeError):
    """Raised when official NAVSIM metrics are requested but unavailable."""


class OfficialNAVSIMWrapper(Protocol):
    def compute(self, records: Sequence[dict[str, Any]]) -> dict[str, float]:
        """Compute official NAVSIM metrics from real records."""


@dataclass(frozen=True)
class NAVSIMMetricResult:
    metrics: dict[str, float | None]
    dry_run: bool
    message: str


class NAVSIMMetricAdapter:
    """Thin adapter around official NAVSIM metrics, with explicit dry-run behavior."""

    def __init__(
        self,
        official_wrapper: OfficialNAVSIMWrapper | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        self.official_wrapper = official_wrapper
        self.dry_run = dry_run

    def evaluate(self, records: Sequence[dict[str, Any]]) -> NAVSIMMetricResult:
        """Evaluate NAVSIM metrics or return a clearly marked dry-run stub."""
        if self.official_wrapper is not None:
            return NAVSIMMetricResult(
                metrics=self.official_wrapper.compute(records),
                dry_run=False,
                message="official NAVSIM wrapper result",
            )

        if self.dry_run:
            return NAVSIMMetricResult(
                metrics={
                    "EPDMS": None,
                    "NC": None,
                    "DAC": None,
                    "DDC": None,
                    "TLC": None,
                    "EP": None,
                    "TTC": None,
                    "LK": None,
                    "HC": None,
                    "EC": None,
                },
                dry_run=True,
                message=(
                    "dry-run stub only: official NAVSIM wrapper is unavailable; "
                    "no real EPDMS or benchmark metric is generated"
                ),
            )

        raise NAVSIMMetricUnavailableError(
            "Official NAVSIM metric wrapper is unavailable. "
            "Set dry_run=True for a non-benchmark stub, or provide an official wrapper. "
            "This adapter will not fabricate EPDMS."
        )
