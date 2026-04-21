"""CLI scaffold for shared-candidate evaluation."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coordiworld.cli.evaluate",
        description="Run evaluation (skeleton only; not implemented).",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--split", default="val", help="Dataset split placeholder.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print("evaluate: not implemented yet")
    print(f"config={args.config}, split={args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
