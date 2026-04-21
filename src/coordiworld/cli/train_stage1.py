"""CLI scaffold for Stage I training."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coordiworld.cli.train_stage1",
        description="Run Stage I training entry point (synthetic smoke only in this repo).",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--epochs", type=int, default=1, help="Placeholder epoch count.")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Use scripts/run_stage1_synthetic.sh for the supported smoke path.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print("train_stage1: full real-data training loop is not implemented in this repository.")
    print("Use bash scripts/run_stage1_synthetic.sh for CPU synthetic smoke training.")
    print(f"config={args.config}, epochs={args.epochs}, synthetic_smoke={args.synthetic_smoke}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
