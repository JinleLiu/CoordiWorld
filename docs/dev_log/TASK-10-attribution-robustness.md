# TASK-10 attribution robustness

## 日期
2026-04-21

## 本轮目标
实现 post-hoc entity attribution、robustness perturbations 和 auditability metrics，不改变主 ranking score 定义，不生成真实实验结果。

## 完成内容
1. 新增 `src/coordiworld/attribution/masking.py`，实现 entity token masking 和 nearby/risk-heuristic entity selection。
2. 新增 `src/coordiworld/attribution/counterfactual.py`，实现 selected trajectory 的 mock-scorer counterfactual `J` 重算和 `delta_i` attribution。
3. 新增 `src/coordiworld/evaluation/robustness.py`，实现 confidence noise、provenance masking、evidence dropout 和 ranking stability。
4. 新增 `src/coordiworld/evaluation/auditability.py`，实现 `RiskDrop@K` 和 `EntityRecall@K`。
5. 新增 `tests/test_attribution.py` 与 `tests/test_robustness.py`，使用 synthetic tokens 和 mock scorer 验证接口。

## 范围控制
- attribution 仅用于 audit-only，不参与主 ranking score。
- 未改变 `CandidateRiskScore` 或 final `J` 定义。
- 未运行真实模型。
- 未读取真实数据。
- 未生成真实实验结果。

## 验收命令
```bash
pytest tests/test_attribution.py tests/test_robustness.py -q
```

## 已知限制
- counterfactual scorer 由调用方注入，本轮只提供 mock-scorer friendly 接口。
- robustness perturbation 作用于 token 副本，不包含完整评估流水线。
- nearby entity selection 使用 deterministic distance/risk heuristic，不是学习模型。
