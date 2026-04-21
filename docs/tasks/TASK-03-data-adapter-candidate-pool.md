# TASK-03: data adapter + candidate pool

## Goal
实现 BaseScenarioSample adapter 接口与 shared-candidate pool 生成模块（synthetic 优先）。

## Allowed files
- `src/coordiworld/data/base_sample.py`
- `src/coordiworld/data/adapters/navsim_adapter.py`
- `src/coordiworld/data/adapters/openscene_adapter.py`
- `src/coordiworld/candidates/pool_builder.py`
- `tests/test_data_adapter_candidate_pool.py`

## Read first
- `docs/specs/DATA_ADAPTER_SPEC.md`
- `docs/specs/EXPERIMENT_PROTOCOL.md`

## Out of scope
- 真实 benchmark 打分
- 训练代码

## Implementation details
- adapter 仅实现接口和 stub，路径来自 `DATA_ROOT/NAVSIM_ROOT/OPENSCENE_ROOT`。
- candidate pool 支持 nominal/speed-scaled/lateral-shift/curvature-perturbed。
- 记录 deterministic seed 与 pool metadata。

## Acceptance criteria
- `pytest -q tests/test_data_adapter_candidate_pool.py` 通过。
- 缺失环境变量时报明确异常。

## Suggested branch name
- `task/03-data-adapter-candidate-pool`

## Whether GPU is required
No.

## Whether real data is required
No（可选用于手工联调）。
