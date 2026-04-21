# TASK-05: rollout model

## Goal
实现 candidate-conditioned structured rollout 模型骨架与输出头接口。

## Allowed files
- `src/coordiworld/model/rollout_model.py`
- `src/coordiworld/model/heads.py`
- `tests/test_rollout_model.py`

## Read first
- `docs/specs/COORDIWORLD_SPEC.md`
- `docs/specs/SCENESUMMARY_SPEC.md`

## Out of scope
- ranking fine-tuning
- calibration 拟合

## Implementation details
- ego future 由 candidate 直接条件化。
- tracked agents 输出 residual motion、existence probability、position covariance。
- 接口先支持 synthetic 前向与 shape 校验。

## Acceptance criteria
- `pytest -q tests/test_rollout_model.py` 通过。
- dry-run 前向在 CPU 上可运行。

## Suggested branch name
- `task/05-rollout-model`

## Whether GPU is required
No（建议支持 CPU）。

## Whether real data is required
No.
