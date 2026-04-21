"""CLI scaffold for shared-candidate evaluation."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coordiworld.cli.evaluate",
        description="Run shared-candidate evaluation entry point (synthetic dry-run only here).",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--split", default="val", help="Dataset split placeholder.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print("evaluate: full real-data evaluation runner is not implemented in this CLI.")
    print("Use bash scripts/run_eval_synthetic.sh for synthetic diagnostics dry-run.")
    print(f"config={args.config}, split={args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
