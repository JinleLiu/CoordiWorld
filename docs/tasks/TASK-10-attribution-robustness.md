# TASK-10: attribution + robustness

## Goal
实现实体级 counterfactual attribution 与 robustness 评估接口（审计用途）。

## Allowed files
- `src/coordiworld/eval/attribution.py`
- `src/coordiworld/eval/robustness.py`
- `tests/test_attribution_robustness.py`

## Read first
- `docs/specs/COORDIWORLD_SPEC.md`
- `docs/specs/METRICS_SPEC.md`
- `docs/specs/EXPERIMENT_PROTOCOL.md`

## Out of scope
- 将 attribution 用作排序输入
- 真实 benchmark 结论

## Implementation details
- 支持 counterfactual entity masking。
- 计算 RiskDrop@K、EntityRecall@K 与稳定性指标。
- robustness 覆盖 confidence noise/provenance masking/evidence dropout。

## Acceptance criteria
- `pytest -q tests/test_attribution_robustness.py` 通过。
- 文档与代码中明确 attribution 为 audit-only。

## Suggested branch name
- `task/10-attribution-robustness`

## Whether GPU is required
No.

## Whether real data is required
No.
