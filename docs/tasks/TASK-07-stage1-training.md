# TASK-07: stage1 training

## Goal
实现 Stage I（structured rollout pretraining）训练接口与配置，不做真实大规模训练。

## Allowed files
- `src/coordiworld/training/stage1.py`
- `src/coordiworld/training/losses_stage1.py`
- `tests/test_stage1_training_dryrun.py`

## Read first
- `docs/specs/COORDIWORLD_SPEC.md`
- `docs/specs/EXPERIMENT_PROTOCOL.md`

## Out of scope
- Stage II ranking
- 真实数据实验结论

## Implementation details
- 覆盖 residual/existence/covariance 相关 loss。
- 提供 synthetic mini-batch dry-run。
- 日志中明确“non-benchmark / synthetic-only”。

## Acceptance criteria
- `pytest -q tests/test_stage1_training_dryrun.py` 通过。
- 单步 dry-run train step 可执行。

## Suggested branch name
- `task/07-stage1-training`

## Whether GPU is required
No（可选加速）。

## Whether real data is required
No.
