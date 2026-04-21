"""Smoke commands advertised by README should remain runnable without real data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_readme_synthetic_validate_command_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coordiworld.cli.validate_data",
            "--dataset",
            "synthetic",
            "--max-samples",
            "2",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_readme_jsonl_validate_command_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coordiworld.cli.validate_data",
            "--dataset",
            "jsonl",
            "--config",
            "configs/datasets/jsonl_example.yaml",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
