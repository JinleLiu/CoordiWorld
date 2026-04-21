"""NAVSIM adapter interface stubs.

This module intentionally does not implement real NAVSIM loading or metrics.
It only defines the environment-variable based interface expected by later
integration work.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from coordiworld.data.base import BaseScenarioSample


@dataclass(frozen=True)
class NavsimAdapterConfig:
    root_env_var: str = "NAVSIM_ROOT"


class NavsimAdapter:
    """Interface placeholder for deterministic NAVSIM -> BaseScenarioSample conversion."""

    def __init__(self, config: NavsimAdapterConfig | None = None) -> None:
        self.config = config or NavsimAdapterConfig()

    def root_from_env(self, env: Mapping[str, str] | None = None) -> Path:
        values = os.environ if env is None else env
        raw_root = values.get(self.config.root_env_var, "").strip()
        if not raw_root:
            raise RuntimeError(
                f"{self.config.root_env_var} is required for NavsimAdapter. "
                "Set it to the NAVSIM dataset root outside git-tracked files."
            )
        return Path(raw_root).expanduser()

    def iter_samples(self, split: str) -> Iterator[BaseScenarioSample]:
        root = self.root_from_env()
        raise NotImplementedError(
            "NavsimAdapter.iter_samples is an interface stub. "
            f"TODO: load split={split!r} from {self.config.root_env_var}={root}, "
            "convert records to BaseScenarioSample, and validate units/fields. "
            "This stub does not read real NAVSIM files or compute NAVSIM metrics."
        )
