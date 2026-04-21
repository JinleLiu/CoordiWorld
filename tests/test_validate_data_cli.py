"""Subprocess tests for validate_data CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "coordiworld.cli.validate_data", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_validate_data_synthetic_passes() -> None:
    result = run_cli("--dataset", "synthetic", "--max-samples", "2")

    assert result.returncode == 0, result.stderr
    assert '"validated_samples": 2' in result.stdout


def test_validate_data_jsonl_example_passes() -> None:
    result = run_cli(
        "--dataset",
        "jsonl",
        "--config",
        "configs/datasets/jsonl_example.yaml",
        "--max-samples",
        "2",
    )

    assert result.returncode == 0, result.stderr
    assert '"dataset": "jsonl"' in result.stdout
    assert '"validated_samples": 1' in result.stdout


def test_validate_data_real_dataset_missing_root_fails(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing-navsim"

    result = run_cli("--dataset", "navsim", "--root", str(missing_root), "--max-samples", "1")

    assert result.returncode != 0
    assert "DataRootError" in result.stderr
    assert "does not exist" in result.stderr
