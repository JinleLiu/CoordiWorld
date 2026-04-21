# TASK-06: risk evaluator

## Goal
实现风险分解与总分 `J(tau)` 聚合接口（collision/rule/uncertainty + calibration hook）。

## Allowed files
- `src/coordiworld/evaluator/risk_heads.py`
- `src/coordiworld/evaluator/score_aggregator.py`
- `tests/test_risk_evaluator.py`

## Read first
- `docs/specs/COORDIWORLD_SPEC.md`
- `docs/specs/METRICS_SPEC.md`

## Out of scope
- attribution
- 训练脚本

## Implementation details
- 独立 risk head 输出可审计子分数。
- `J(tau)` 聚合与 calibrator 接口解耦。
- 明确区分 pre-calibration 与 post-calibration 分数。

## Acceptance criteria
- `pytest -q tests/test_risk_evaluator.py` 通过。
- synthetic 输入下各风险分量可解释且数值稳定。

## Suggested branch name
- `task/06-risk-evaluator`

## Whether GPU is required
No.

## Whether real data is required
No.
