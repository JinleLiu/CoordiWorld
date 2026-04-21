"""CLI scaffold for ablation experiments."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coordiworld.cli.run_ablation",
        description="Run ablations (skeleton only; not implemented).",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--ablation", default="none", help="Ablation switch placeholder.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print("run_ablation: not implemented yet")
    print(f"config={args.config}, ablation={args.ablation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
