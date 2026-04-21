#!/usr/bin/env python3
"""Check CoordiWorld local development environment variables."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

PATH_ENV_VARS: tuple[str, ...] = (
    "DATA_ROOT",
    "NAVSIM_ROOT",
    "OPENSCE_ROOT",
    "NUSCENES_ROOT",
    "WAYMO_ROOT",
    "OUTPUT_ROOT",
    "CHECKPOINT_ROOT",
)
NON_PATH_ENV_VARS: tuple[str, ...] = ("WANDB_MODE",)
REQUIRED_ENV_VARS: tuple[str, ...] = (*PATH_ENV_VARS, *NON_PATH_ENV_VARS)
SUPPORTED_WANDB_MODES: frozenset[str] = frozenset({"online", "offline", "disabled", "dryrun"})


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines from an env file without external dependencies."""
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_number}: empty environment variable name")
        values[key] = value.strip().strip("'\"")
    return values


def build_environment(base_env: Mapping[str, str], env_file: Path | None) -> dict[str, str]:
    env: dict[str, str] = {}
    if env_file is not None:
        env.update(parse_env_file(env_file))
    env.update(base_env)
    return env


def check_path_var(name: str, env: Mapping[str, str]) -> CheckResult:
    value = env.get(name, "").strip()
    if not value:
        return CheckResult(name=name, ok=False, detail="missing")

    path = Path(value).expanduser()
    if not path.exists():
        return CheckResult(name=name, ok=False, detail=f"{path} does not exist")
    if not path.is_dir():
        return CheckResult(name=name, ok=False, detail=f"{path} is not a directory")
    if not os.access(path, os.R_OK | os.X_OK):
        return CheckResult(name=name, ok=False, detail=f"{path} is not readable/traversable")
    return CheckResult(name=name, ok=True, detail=str(path))


def check_wandb_mode(env: Mapping[str, str]) -> CheckResult:
    value = env.get("WANDB_MODE", "").strip()
    if not value:
        return CheckResult(name="WANDB_MODE", ok=False, detail="missing")
    if value not in SUPPORTED_WANDB_MODES:
        modes = ", ".join(sorted(SUPPORTED_WANDB_MODES))
        return CheckResult(name="WANDB_MODE", ok=False, detail=f"{value!r} not in {{{modes}}}")
    return CheckResult(name="WANDB_MODE", ok=True, detail=value)


def run_checks(env: Mapping[str, str]) -> list[CheckResult]:
    results = [check_path_var(name, env) for name in PATH_ENV_VARS]
    results.append(check_wandb_mode(env))
    return results


def print_results(results: Sequence[CheckResult]) -> None:
    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check CoordiWorld environment variables and directory accessibility. "
            "This command does not create directories, download datasets, read dataset files, "
            "or call GPU APIs."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional private env file with KEY=VALUE lines. Shell environment overrides it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    env_file = args.env_file
    if env_file is not None and not env_file.is_file():
        parser.error(f"--env-file does not exist or is not a file: {env_file}")

    env = build_environment(os.environ, env_file)
    results = run_checks(env)
    print_results(results)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
