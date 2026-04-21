# TASK-01: scene summary schema

## Goal
定义 SceneSummary 的强类型 schema/验证器，覆盖顶层字段与 agent 字段约束。

## Allowed files
- `src/coordiworld/scene_summary/schema.py`
- `src/coordiworld/scene_summary/validators.py`
- `tests/test_scene_summary_schema.py`

## Read first
- `docs/specs/SCENESUMMARY_SPEC.md`
- `docs/specs/COORDIWORLD_SPEC.md`

## Out of scope
- ICA 生成器实现
- 任何 LLM 调用

## Implementation details
- 为 `scene_id/timestamp/coordinate_frame/ego/agents/map_tokens/provenance/metadata` 建立 schema。
- 为 agent 字段提供单位/范围/缺失值校验。
- 产出 deterministic 序列化顺序（便于审计与回归测试）。

## Acceptance criteria
- `pytest -q tests/test_scene_summary_schema.py` 通过。
- 提供 synthetic fixture 验证合法与非法样本。

## Suggested branch name
- `task/01-scene-summary-schema`

## Whether GPU is required
No.

## Whether real data is required
No.
