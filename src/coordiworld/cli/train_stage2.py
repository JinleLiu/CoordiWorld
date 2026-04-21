"""CLI scaffold for Stage II training."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coordiworld.cli.train_stage2",
        description="Run Stage II training (skeleton only; not implemented).",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--pairs-per-batch", type=int, default=32, help="Placeholder batch size.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print("train_stage2: not implemented yet")
    print(f"config={args.config}, pairs_per_batch={args.pairs_per_batch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
