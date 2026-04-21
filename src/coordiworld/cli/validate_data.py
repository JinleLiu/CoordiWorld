"""Validate CoordiWorld dataset adapters without reading hidden project data dirs."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from coordiworld.data.base import (
    BaseScenarioSample,
    DataRootError,
    DatasetFormatError,
    MissingDependencyError,
    validate_base_scenario_sample,
)
from coordiworld.data.registry import available_datasets, build_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coordiworld.cli.validate_data",
        description="Validate synthetic, JSONL, or real-data adapter configuration.",
    )
    parser.add_argument(
        "--dataset",
        choices=available_datasets(),
        default="synthetic",
        help="Dataset adapter name.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Dataset config YAML/JSON.")
    parser.add_argument("--root", default=None, help="Override real dataset root.")
    parser.add_argument("--split", default=None, help="Dataset split override.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to check.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config_file(args.config)
    config["dataset"] = args.dataset
    if args.root is not None:
        config["root"] = args.root
    if args.split is not None:
        config["split"] = args.split
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples

    try:
        dataset = build_dataset(args.dataset, config)
        checked = validate_dataset_samples(
            dataset,
            split=str(config.get("split", "synthetic" if args.dataset == "synthetic" else "val")),
            max_samples=args.max_samples or _optional_int(config.get("max_samples")) or 1,
        )
    except (
        DataRootError,
        DatasetFormatError,
        MissingDependencyError,
        NotImplementedError,
        KeyError,
        ValueError,
    ) as error:
        print(f"validate_data failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "config": str(args.config) if args.config else None,
                "validated_samples": checked,
                "real_data": args.dataset not in {"synthetic", "jsonl"},
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def validate_dataset_samples(dataset: object, *, split: str, max_samples: int) -> int:
    if max_samples <= 0:
        raise ValueError("max_samples must be > 0")
    checked = 0
    for sample in iter_dataset_samples(dataset, split=split):
        validate_base_scenario_sample(sample)
        checked += 1
        if checked >= max_samples:
            break
    if checked == 0:
        raise DatasetFormatError(f"dataset yielded no samples for split={split!r}")
    return checked


def iter_dataset_samples(dataset: object, *, split: str) -> Iterator[BaseScenarioSample]:
    iter_samples = getattr(dataset, "iter_samples", None)
    if callable(iter_samples):
        yield from iter_samples(split)
        return
    length = len(dataset)  # type: ignore[arg-type]
    for index in range(length):
        yield dataset[index]  # type: ignore[index]


def load_config_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise DatasetFormatError(f"config file does not exist: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise DatasetFormatError(f"config JSON must be an object: {path}")
        return data
    return load_simple_yaml(path)


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the small YAML subset used by this repository's config templates."""
    result: dict[str, Any] = {}
    current_mapping: dict[str, Any] | None = None
    current_indent = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, raw_value = raw_line.strip().partition(":")
        if separator != ":":
            raise DatasetFormatError(f"unsupported config line in {path}: {raw_line!r}")
        if raw_value.strip() == "":
            mapping: dict[str, Any] = {}
            result[key] = mapping
            current_mapping = mapping
            current_indent = indent
            continue
        target = result
        if current_mapping is not None and indent > current_indent:
            target = current_mapping
        elif indent <= current_indent:
            current_mapping = None
        target[key] = parse_scalar(raw_value.strip())
    return result


def parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
