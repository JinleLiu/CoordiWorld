# TASK-09: calibration evaluation

## Goal
实现 post-hoc calibration 与评估模块（ECE/Brier + reliability diagnostics）。

## Allowed files
- `src/coordiworld/eval/calibration.py`
- `src/coordiworld/eval/reliability.py`
- `tests/test_calibration_evaluation.py`

## Read first
- `docs/specs/METRICS_SPEC.md`
- `docs/specs/EXPERIMENT_PROTOCOL.md`

## Out of scope
- 模型主干实现
- attribution 逻辑

## Implementation details
- 预留 isotonic/temperature/platt 等可插拔接口。
- 输出 pre/post calibration 对比指标。
- 无官方 wrapper 时使用 stub adapter + synthetic 校验。

## Acceptance criteria
- `pytest -q tests/test_calibration_evaluation.py` 通过。
- synthetic 数据下 ECE/Brier 计算可复现。

## Suggested branch name
- `task/09-calibration-evaluation`

## Whether GPU is required
No.

## Whether real data is required
No.
