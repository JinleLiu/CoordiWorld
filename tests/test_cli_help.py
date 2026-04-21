"""CLI --help contract tests for scaffold commands."""

import subprocess
import sys

COMMANDS = [
    ["-m", "coordiworld.cli.build_scene_summary", "--help"],
    ["-m", "coordiworld.cli.train_stage1", "--help"],
    ["-m", "coordiworld.cli.train_stage2", "--help"],
    ["-m", "coordiworld.cli.calibrate", "--help"],
    ["-m", "coordiworld.cli.evaluate", "--help"],
    ["-m", "coordiworld.cli.run_ablation", "--help"],
]


def test_cli_help_commands() -> None:
    for args in COMMANDS:
        result = subprocess.run(
            [sys.executable, *args],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
