# TASK-08: stage2 ranking

## Goal
实现 Stage II（pairwise ranking fine-tuning）训练接口，基于 shared-candidate protocol。

## Allowed files
- `src/coordiworld/training/stage2.py`
- `src/coordiworld/training/losses_stage2.py`
- `tests/test_stage2_ranking_dryrun.py`

## Read first
- `docs/specs/COORDIWORLD_SPEC.md`
- `docs/specs/EXPERIMENT_PROTOCOL.md`
- `docs/specs/METRICS_SPEC.md`

## Out of scope
- calibration 拟合
- attribution 评估

## Implementation details
- pairwise ranking loss 接口化。
- 输入使用固定候选池并记录候选生成元信息。
- dry-run 支持 synthetic pair sampling。

## Acceptance criteria
- `pytest -q tests/test_stage2_ranking_dryrun.py` 通过。
- 训练步可输出 ranking loss 与基础相关性诊断。

## Suggested branch name
- `task/08-stage2-ranking`

## Whether GPU is required
No（可选）。

## Whether real data is required
No.
