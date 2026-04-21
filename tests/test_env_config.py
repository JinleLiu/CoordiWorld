"""Environment configuration smoke tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENV_KEYS = {
    "DATA_ROOT",
    "NAVSIM_ROOT",
    "OPENSCE_ROOT",
    "NUSCENES_ROOT",
    "WAYMO_ROOT",
    "OUTPUT_ROOT",
    "CHECKPOINT_ROOT",
    "WANDB_MODE",
}


def parse_env_example(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        assert separator == "=", f"Expected KEY=VALUE line in {path}: {raw_line!r}"
        values[key.strip()] = value.strip()
    return values


def test_env_example_contains_required_keys() -> None:
    env_example = parse_env_example(REPO_ROOT / ".env.example")

    missing = REQUIRED_ENV_KEYS.difference(env_example)

    assert not missing, f".env.example missing keys: {sorted(missing)}"


def test_check_env_declares_same_required_keys() -> None:
    module_path = REPO_ROOT / "scripts" / "check_env.py"
    spec = importlib.util.spec_from_file_location("check_env", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    declared = set(module.REQUIRED_ENV_VARS)

    assert REQUIRED_ENV_KEYS == declared
