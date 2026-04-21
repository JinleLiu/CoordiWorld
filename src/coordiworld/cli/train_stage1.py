"""CLI scaffold for Stage I training."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coordiworld.cli.train_stage1",
        description="Run Stage I training (skeleton only; not implemented).",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--epochs", type=int, default=1, help="Placeholder epoch count.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print("train_stage1: not implemented yet")
    print(f"config={args.config}, epochs={args.epochs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
