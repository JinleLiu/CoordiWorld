"""JSON and dict I/O for SceneSummary dataclasses."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, TypeVar

from coordiworld.scene_summary.schema import AgentState, EgoState, MapToken, SceneSummary

DataclassT = TypeVar("DataclassT", EgoState, AgentState, MapToken)


def scene_summary_to_dict(summary: SceneSummary) -> dict[str, Any]:
    """Convert a SceneSummary dataclass tree into plain Python containers."""
    if not isinstance(summary, SceneSummary):
        raise ValueError("summary must be a SceneSummary")
    return asdict(summary)


def scene_summary_from_dict(data: dict[str, Any]) -> SceneSummary:
    """Construct a SceneSummary from a strict dict representation."""
    mapping = _require_mapping(data, "SceneSummary")
    _check_dataclass_keys(SceneSummary, mapping, "SceneSummary")

    ego = _construct_dataclass(EgoState, mapping["ego"], "ego")
    agents = [
        _construct_dataclass(AgentState, value, f"agents[{index}]")
        for index, value in enumerate(_require_list(mapping["agents"], "agents"))
    ]
    map_tokens = [
        _construct_dataclass(MapToken, value, f"map_tokens[{index}]")
        for index, value in enumerate(_require_list(mapping["map_tokens"], "map_tokens"))
    ]

    return SceneSummary(
        scene_id=mapping["scene_id"],
        timestamp=mapping["timestamp"],
        coordinate_frame=mapping["coordinate_frame"],
        ego=ego,
        agents=agents,
        map_tokens=map_tokens,
        provenance=mapping["provenance"],
        metadata=mapping["metadata"],
    )


def scene_summary_to_json(summary: SceneSummary, indent: int | None = None) -> str:
    """Serialize a SceneSummary to strict JSON."""
    return json.dumps(
        scene_summary_to_dict(summary),
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
    )


def scene_summary_from_json(text: str) -> SceneSummary:
    """Deserialize strict JSON text into a SceneSummary."""
    try:
        data = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid SceneSummary JSON: {error}") from error
    return scene_summary_from_dict(data)


def save_scene_summary_json(summary: SceneSummary, path: str | Path) -> None:
    """Write a SceneSummary JSON file without creating directories or reading datasets."""
    output_path = Path(path)
    output_path.write_text(scene_summary_to_json(summary, indent=2) + "\n", encoding="utf-8")


def load_scene_summary_json(path: str | Path) -> SceneSummary:
    """Read a SceneSummary JSON file."""
    input_path = Path(path)
    return scene_summary_from_json(input_path.read_text(encoding="utf-8"))


def _construct_dataclass(cls: type[DataclassT], value: Any, path: str) -> DataclassT:
    mapping = _require_mapping(value, path)
    _check_dataclass_keys(cls, mapping, path)
    kwargs = {field.name: mapping[field.name] for field in fields(cls)}
    return cls(**kwargs)


def _check_dataclass_keys(cls: type[Any], mapping: dict[str, Any], path: str) -> None:
    expected = {field.name for field in fields(cls)}
    actual = set(mapping)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise ValueError(f"{path} missing required fields: {missing}")
    if unexpected:
        raise ValueError(f"{path} has unexpected fields: {unexpected}")


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a dict")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON numeric constant: {value}")
