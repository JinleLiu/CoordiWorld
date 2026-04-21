"""CLI scaffold for building SceneSummary artifacts."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coordiworld.cli.build_scene_summary",
        description="Build SceneSummary (skeleton only; not implemented).",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate args without running work.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print("build_scene_summary: not implemented yet")
    print(f"config={args.config}, dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
