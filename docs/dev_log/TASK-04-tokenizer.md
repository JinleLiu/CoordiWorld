# TASK-04 tokenizer

## 日期
2026-04-21

## 本轮目标
实现 `SceneSummary` 与 shared candidate trajectory 的 tokenizer，不实现 rollout model、训练或评估。

## 完成内容
1. 新增 `src/coordiworld/tokens/scene_tokenizer.py`，输出 ego tensor、agent tensor、map tensor、masks、local slot remapping 与 feature metadata。
2. 新增 `src/coordiworld/tokens/map_tokenizer.py`，实现最多 24 个 map tokens 的距离/规则风险启发式筛选与 padding。
3. 新增 `src/coordiworld/tokens/action_tokenizer.py`，将 candidate trajectories `[M,H,3]` 转成 action token tensor。
4. 新增 `tests/test_tokenizer.py`，覆盖 shape、mask、agent/map truncation、clip-local slot remapping 和 action delta token。

## 范围控制
- 未实现 rollout model。
- 未实现训练。
- 未实现风险评估。
- 未读取真实数据。
- 未调用 GPU。
- 未引入 torch、numpy 或新的默认依赖。

## 验收命令
```bash
pytest tests/test_tokenizer.py -q
```

## 已知限制
- 当前 tensor 使用嵌套 Python list，后续接入模型时可在 optional torch/numpy dependency 下转换。
- agent/map 风险排序是 deterministic heuristic，不是学习模型。
- tokenizer 依赖 `TASK-01` 的 `SceneSummary` schema。
