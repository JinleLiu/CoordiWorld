# TASK-00: repo scaffold

## Goal
建立最小工程骨架与配置入口，使后续任务可在不依赖真实数据的前提下执行 dry-run 和 pytest 验收。

## Allowed files
- `src/coordiworld/__init__.py`
- `src/coordiworld/config/*.py`
- `tests/test_repo_scaffold.py`
- `pyproject.toml`（仅最小依赖/工具配置）

## Read first
- `docs/specs/COORDIWORLD_SPEC.md`
- `docs/specs/DATA_ADAPTER_SPEC.md`
- `docs/specs/EXPERIMENT_PROTOCOL.md`

## Out of scope
- 任何模型实现
- 任何训练脚本
- 任何真实数据读取

## Implementation details
- 建立包结构与统一配置对象。
- 提供 dry-run 命令入口（仅打印配置与校验项）。
- 将数据路径读取统一到环境变量层。

## Acceptance criteria
- `pytest -q tests/test_repo_scaffold.py` 通过。
- dry-run 命令在无真实数据下可执行并输出明确提示。

## Suggested branch name
- `task/00-repo-scaffold`

## Whether GPU is required
No.

## Whether real data is required
No.
