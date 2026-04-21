# TASK-01 scene summary schema

## 日期
2026-04-21

## 本轮目标
实现 CoordiWorld 的结构化 `SceneSummary` schema、JSON I/O 与 validator，作为后续 tokenizer、rollout model、risk evaluator 和 ICA generator 的统一接口。

## 完成内容
1. 新增 `src/coordiworld/scene_summary/schema.py`，定义 `EgoState`、`AgentState`、`MapToken`、`SceneSummary` dataclass。
2. 新增 `src/coordiworld/scene_summary/validators.py`，实现确定性的结构与数值约束校验。
3. 新增 `src/coordiworld/scene_summary/io.py`，提供 dict/JSON roundtrip 与文件读写。
4. 更新 `src/coordiworld/scene_summary/__init__.py`，导出 schema、validator 与 I/O API。
5. 新增 `tests/test_scene_summary_schema.py`，使用 synthetic fixture 覆盖合法样例、非法字段与 JSON 文件 roundtrip。

## 范围控制
- 未实现 ICA fusion。
- 未实现 CoordiWorld model。
- 未实现 tokenizer。
- 未实现训练或评估。
- 未读取真实数据。
- 未调用 GPU。

## 验收命令
```bash
python -m pytest tests/test_scene_summary_schema.py -q
python -m pytest tests/test_imports.py tests/test_cli_help.py tests/test_env_config.py -q
ruff check src tests
```

## 已知限制
- 当前 schema 只定义单帧结构化世界状态，不包含历史窗口容器。
- `MapToken.type` 只支持本轮任务列出的五类合法值。
- I/O 层执行严格字段匹配，但不会替代 `validate_scene_summary` 的完整语义校验。
