"""CLI for SceneSummary dry-run/smoke validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coordiworld.cli.validate_data import load_config_file
from coordiworld.data.base import DataRootError, DatasetFormatError, MissingDependencyError
from coordiworld.data.registry import available_datasets, build_dataset
from coordiworld.scene_summary.io import scene_summary_to_dict
from coordiworld.scene_summary.validators import validate_scene_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coordiworld.cli.build_scene_summary",
        description="Validate or dry-run SceneSummary construction from dataset adapters.",
    )
    parser.add_argument("--dataset", choices=available_datasets(), default="synthetic")
    parser.add_argument("--config", type=Path, default=None, help="Path to dataset config file.")
    parser.add_argument("--root", default=None, help="Override real dataset root.")
    parser.add_argument("--max-samples", type=int, default=1, help="Smoke sample count.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For native real datasets, validate arguments without claiming conversion.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.dataset not in {"synthetic", "jsonl"} and args.dry_run:
        print(
            "build_scene_summary: dry-run only for native datasets; "
            "real conversion requires official dataset APIs."
        )
        print(f"dataset={args.dataset}, config={args.config}, root={args.root}")
        return 0

    config = load_config_file(args.config)
    config["dataset"] = args.dataset
    if args.root is not None:
        config["root"] = args.root
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples
    try:
        dataset = build_dataset(args.dataset, config)
        sample = next(dataset.iter_samples(str(config.get("split", "synthetic"))))
        summary = sample.scene_summary_history[-1]
        validate_scene_summary(summary)
    except (
        DataRootError,
        DatasetFormatError,
        MissingDependencyError,
        NotImplementedError,
        StopIteration,
        ValueError,
    ) as error:
        print(f"build_scene_summary failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "dry_run": True,
                "dataset": args.dataset,
                "scene_summary": scene_summary_to_dict(summary),
                "note": "Smoke validation only; no real benchmark artifact is generated.",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
