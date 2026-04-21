"""Smoke tests for package import scaffold."""

import importlib

MODULES = [
    "coordiworld",
    "coordiworld.scene_summary",
    "coordiworld.data",
    "coordiworld.tokens",
    "coordiworld.models",
    "coordiworld.risks",
    "coordiworld.attribution",
    "coordiworld.training",
    "coordiworld.evaluation",
    "coordiworld.visualization",
    "coordiworld.cli",
]


def test_imports() -> None:
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        assert module is not None
