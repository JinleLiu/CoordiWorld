# TASK-02 ICA generator

## 日期
2026-04-21

## 本轮目标
实现适配 CoordiWorld 的 deterministic ICA-style `SceneSummary` generator，覆盖多源几何种子、radar 速度附加、camera 投影视觉语义附加与 provenance/ambiguity 记录。

## 完成内容
1. 新增 `src/coordiworld/scene_summary/transforms.py`，提供 sensor-to-ego、radar polar to ego-BEV、camera projection 与 bbox helper。
2. 新增 `src/coordiworld/scene_summary/association.py`，提供 BEV gating、Mahalanobis fallback、class compatibility 与 deterministic assignment。
3. 新增 `src/coordiworld/scene_summary/fusion.py`，提供连续字段加权融合、类别投票、语义合并、ambiguity flag 与 fusion trace。
4. 新增 `src/coordiworld/scene_summary/generator.py`，实现 `MultiSourceFacts -> SceneSummary` 的 ICA-style 流程。
5. 新增 `tests/test_ica_fusion.py`，使用 synthetic cases 覆盖 LiDAR/BEVFusion 冲突融合、radar 速度附加、camera 语义附加、camera-only 不导出。

## 范围控制
- 未实现 CoordiWorld model。
- 未实现 tokenizer。
- 未实现训练或风险评估。
- 未读取真实数据。
- 未调用 GPU。
- 未调用任何 LLM 服务。

## 验收命令
```bash
pytest tests/test_ica_fusion.py -q
```

## 已知限制
- assignment 使用纯标准库 exact enumerator，适合当前 synthetic smoke tests，不面向大规模生产匹配。
- camera projection helper 只提供针孔投影接口，不包含真实相机标定加载。
- 当前 generator 依赖 `TASK-01` 的 `SceneSummary` schema 与 validator。
