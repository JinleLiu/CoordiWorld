"""CLI scaffold for Stage II training."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coordiworld.cli.train_stage2",
        description="Run Stage II ranking entry point (schema dry-run only in this repo).",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--pairs-per-batch", type=int, default=32, help="Placeholder batch size.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print("train_stage2: full pairwise ranking training is not implemented in this repository.")
    print("Use bash scripts/run_stage2_synthetic.sh for pairwise batch schema dry-run.")
    print(f"config={args.config}, pairs_per_batch={args.pairs_per_batch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
