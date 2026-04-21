"""CLI scaffold for post-hoc calibration."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coordiworld.cli.calibrate",
        description="Run calibration (skeleton only; not implemented).",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file.")
    parser.add_argument("--method", default="temperature", help="Placeholder calibration method.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    print("calibrate: not implemented yet")
    print(f"config={args.config}, method={args.method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
