"""Waymo Open Dataset adapter boundary for optional SceneSummary/ICA sources."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from coordiworld.data.base import BaseScenarioSample, MissingDependencyError, resolve_data_root


@dataclass(frozen=True)
class WaymoAdapterConfig:
    root: str | Path | None = None
    split: str = "val"
    max_samples: int | None = None
    root_env_var: str = "WAYMO_ROOT"
    official_module: str = "waymo_open_dataset"

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "WaymoAdapterConfig":
        mapping = data or {}
        return cls(
            root=mapping.get("root"),
            split=str(mapping.get("split", cls.split)),
            max_samples=_optional_int(mapping.get("max_samples")),
        )


class WaymoAdapter:
    """Waymo Open Dataset -> SceneSummary/BaseScenarioSample interface stub."""

    def __init__(self, config: WaymoAdapterConfig | None = None) -> None:
        self.config = config or WaymoAdapterConfig()

    def validate_root(self) -> Path:
        return resolve_data_root(
            explicit_root=self.config.root,
            env_var=self.config.root_env_var,
            dataset_name="Waymo",
        )

    def __len__(self) -> int:
        self._validate_ready()
        raise NotImplementedError("Native Waymo sample indexing is not implemented yet.")

    def __getitem__(self, index: int) -> BaseScenarioSample:
        if index < 0:
            raise IndexError(index)
        self._validate_ready()
        raise NotImplementedError(
            "Native Waymo __getitem__ is not implemented yet. "
            "TODO: convert official Waymo records into SceneSummary/BaseScenarioSample."
        )

    def iter_samples(self, split: str | None = None) -> Iterator[BaseScenarioSample]:
        self._validate_ready()
        resolved_split = split or self.config.split
        raise NotImplementedError(
            "WaymoAdapter.iter_samples is a native-data TODO. "
            f"Root is validated for split={resolved_split!r}, but this repository will not "
            "fabricate Waymo samples."
        )
        yield  # pragma: no cover

    def _validate_ready(self) -> Path:
        root = self.validate_root()
        if find_spec(self.config.official_module) is None:
            raise MissingDependencyError(
                "Waymo Open Dataset dependency is unavailable. Install/configure the official "
                f"package providing module {self.config.official_module!r}; this repository "
                "does not ship or fake Waymo data."
            )
        return root


def build_waymo_dataset(config: dict[str, Any] | None = None) -> WaymoAdapter:
    return WaymoAdapter(WaymoAdapterConfig.from_mapping(config))


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
