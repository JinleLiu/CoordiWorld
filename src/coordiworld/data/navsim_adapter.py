"""NAVSIM dataset adapter interface.

This adapter intentionally does not fabricate NAVSIM samples. It validates the
dataset root and official dependency boundary, then raises a clear TODO for
native parsing until the official API is wired in.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from coordiworld.data.base import (
    BaseScenarioSample,
    MissingDependencyError,
    resolve_data_root,
)


@dataclass(frozen=True)
class NavsimAdapterConfig:
    root: str | Path | None = None
    split: str = "val"
    max_samples: int | None = None
    root_env_var: str = "NAVSIM_ROOT"
    official_module: str = "navsim"

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "NavsimAdapterConfig":
        mapping = data or {}
        return cls(
            root=mapping.get("root"),
            split=str(mapping.get("split", cls.split)),
            max_samples=_optional_int(mapping.get("max_samples")),
        )


class NavsimAdapter:
    """Interface placeholder for deterministic NAVSIM -> BaseScenarioSample conversion."""

    def __init__(self, config: NavsimAdapterConfig | None = None) -> None:
        self.config = config or NavsimAdapterConfig()

    def validate_root(self) -> Path:
        return resolve_data_root(
            explicit_root=self.config.root,
            env_var=self.config.root_env_var,
            dataset_name="NAVSIM",
        )

    def __len__(self) -> int:
        self._validate_ready()
        raise NotImplementedError("Native NAVSIM sample indexing is not implemented yet.")

    def __getitem__(self, index: int) -> BaseScenarioSample:
        if index < 0:
            raise IndexError(index)
        self._validate_ready()
        raise NotImplementedError(
            "Native NAVSIM __getitem__ is not implemented yet. "
            "TODO: convert official NAVSIM records into BaseScenarioSample."
        )

    def iter_samples(self, split: str | None = None) -> Iterator[BaseScenarioSample]:
        self._validate_ready()
        resolved_split = split or self.config.split
        raise NotImplementedError(
            "NavsimAdapter.iter_samples is a native-data TODO. "
            f"Official NAVSIM root is validated for split={resolved_split!r}, but this "
            "repository will not fabricate NAVSIM samples."
        )
        yield  # pragma: no cover

    def _validate_ready(self) -> Path:
        root = self.validate_root()
        if find_spec(self.config.official_module) is None:
            raise MissingDependencyError(
                "Official NAVSIM dependency is unavailable. Install/configure the official "
                f"NAVSIM package providing module {self.config.official_module!r}; "
                "this repository does not ship or fake NAVSIM data."
            )
        return root


NAVSIMScenarioDataset = NavsimAdapter


def build_navsim_dataset(config: dict[str, Any] | None = None) -> NavsimAdapter:
    return NavsimAdapter(NavsimAdapterConfig.from_mapping(config))


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
