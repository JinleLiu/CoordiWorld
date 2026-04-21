# TASK-03 data adapter candidate pool

## 日期
2026-04-21

## 本轮目标
实现 CoordiWorld 的数据接口、synthetic dataset 与 shared candidate pool，不实现模型、训练、风险评估或真实 NAVSIM metric。

## 完成内容
1. 新增 `src/coordiworld/data/base.py`，定义 `BaseScenarioSample`、`ScenarioLabels` 与 `[M,H,3]`/`[H,3]` 形状校验。
2. 新增 `src/coordiworld/data/candidate_pool.py`，支持 nominal、speed-scaled、lateral-shift、curvature-perturbed candidate pool。
3. 新增 `src/coordiworld/data/synthetic.py`，提供 deterministic synthetic samples 和 shared candidate pool。
4. 新增 `src/coordiworld/data/navsim_adapter.py`，只保留环境变量驱动的 NAVSIM adapter 接口与 TODO。
5. 新增 `src/coordiworld/data/collate.py`，提供不依赖 tensor 库的 sample batch collate。
6. 新增 `tests/test_candidate_pool.py` 与 `tests/test_synthetic_dataset.py`，覆盖 candidate pool、synthetic dataset、collate 和 NAVSIM env stub。

## 范围控制
- 未实现 CoordiWorld model。
- 未实现 tokenizer。
- 未实现训练或风险评估。
- 未实现真实 NAVSIM metric。
- 未读取真实数据。
- 未调用 GPU。

## 验收命令
```bash
pytest tests/test_candidate_pool.py tests/test_synthetic_dataset.py -q
```

## 已知限制
- Candidate pool 使用简单运动学扰动生成 synthetic proposals，不代表真实 planner。
- `NavsimAdapter` 只做接口和环境变量检查，真实 NAVSIM record conversion 仍是 TODO。
- 当前数据接口依赖 `TASK-01` 的 `SceneSummary` schema。
