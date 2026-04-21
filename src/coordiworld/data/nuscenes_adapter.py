"""nuScenes adapter boundary for optional SceneSummary/ICA data sources."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from coordiworld.data.base import BaseScenarioSample, MissingDependencyError, resolve_data_root


@dataclass(frozen=True)
class NuScenesAdapterConfig:
    root: str | Path | None = None
    split: str = "val"
    max_samples: int | None = None
    root_env_var: str = "NUSCENES_ROOT"
    official_module: str = "nuscenes"

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "NuScenesAdapterConfig":
        mapping = data or {}
        return cls(
            root=mapping.get("root"),
            split=str(mapping.get("split", cls.split)),
            max_samples=_optional_int(mapping.get("max_samples")),
        )


class NuScenesAdapter:
    """nuScenes -> SceneSummary/BaseScenarioSample interface stub."""

    def __init__(self, config: NuScenesAdapterConfig | None = None) -> None:
        self.config = config or NuScenesAdapterConfig()

    def validate_root(self) -> Path:
        return resolve_data_root(
            explicit_root=self.config.root,
            env_var=self.config.root_env_var,
            dataset_name="nuScenes",
        )

    def __len__(self) -> int:
        self._validate_ready()
        raise NotImplementedError("Native nuScenes sample indexing is not implemented yet.")

    def __getitem__(self, index: int) -> BaseScenarioSample:
        if index < 0:
            raise IndexError(index)
        self._validate_ready()
        raise NotImplementedError(
            "Native nuScenes __getitem__ is not implemented yet. "
            "TODO: convert nuScenes/devkit records into SceneSummary/BaseScenarioSample."
        )

    def iter_samples(self, split: str | None = None) -> Iterator[BaseScenarioSample]:
        self._validate_ready()
        resolved_split = split or self.config.split
        raise NotImplementedError(
            "NuScenesAdapter.iter_samples is a native-data TODO. "
            f"Root is validated for split={resolved_split!r}, but this repository will not "
            "fabricate nuScenes samples."
        )
        yield  # pragma: no cover

    def _validate_ready(self) -> Path:
        root = self.validate_root()
        if find_spec(self.config.official_module) is None:
            raise MissingDependencyError(
                "nuScenes devkit is unavailable. Install/configure the official devkit "
                f"providing module {self.config.official_module!r}; this repository does "
                "not ship or fake nuScenes data."
            )
        return root


def build_nuscenes_dataset(config: dict[str, Any] | None = None) -> NuScenesAdapter:
    return NuScenesAdapter(NuScenesAdapterConfig.from_mapping(config))


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
