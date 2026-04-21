# TASK-06 risk evaluator

## 日期
2026-04-21

## 本轮目标
实现 collision risk、map-grounded rule-violation risk、predictive uncertainty、calibration 接口和 final score `J`，不实现训练循环或论文表格。

## 完成内容
1. 新增 `src/coordiworld/risks/geometry.py`，提供 box geometry、SAT overlap、polyline/polygon、soft margin、smooth max 和 noisy-OR helper。
2. 新增 `src/coordiworld/risks/collision.py`，实现 ego box 与 predicted agent box interaction feature、geometry fallback 和 noisy-OR plan aggregation。
3. 新增 `src/coordiworld/risks/rule_violation.py`，实现 lane departure、drivable area、stop line、traffic light conflict 和 smooth max 聚合。
4. 新增 `src/coordiworld/risks/uncertainty.py`，实现 existence-weighted trace covariance 与 `u95` normalized uncertainty。
5. 新增 `src/coordiworld/risks/calibration.py`，实现 isotonic-style binned calibrator 的 fit/save/load/apply 接口。
6. 新增 `src/coordiworld/risks/scoring.py`，实现 lower-is-better final score `J`。
7. 新增 `tests/test_risk_scoring.py`，覆盖 collision/safe、violation/compliant、uncertainty 和 calibration save/load。
8. 固定 `CandidateRiskScore` 字段契约，避免后续训练/评估接入时无意改破接口。

## 范围控制
- 未实现训练循环。
- 未实现论文表格。
- 未读取真实数据。
- 未调用 GPU。
- 未实现 attribution。

## 验收命令
```bash
pytest tests/test_risk_scoring.py -q
```

## 已知限制
- collision 使用几何 soft fallback，不是学习到的 collision head。
- rule violation 使用 synthetic map token 几何规则，不接入真实地图 API。
- calibration 是轻量 binned isotonic-style 接口，后续可替换为更完整的校准器。
