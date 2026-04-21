# TASK-02: ICA generator

## Goal
实现 deterministic 的 ICA-style SceneSummary generator 流程骨架（接口 + mock/stub）。

## Allowed files
- `src/coordiworld/scene_summary/ica_generator.py`
- `src/coordiworld/scene_summary/fusion_steps/*.py`
- `tests/test_ica_generator.py`

## Read first
- `docs/specs/SCENESUMMARY_SPEC.md`
- `docs/specs/DATA_ADAPTER_SPEC.md`

## Out of scope
- 深度学习检测器训练
- 非确定性随机融合逻辑

## Implementation details
- 分阶段接口：coordinate normalization、geometry seed matching、velocity attachment、semantic attachment、continuous fusion、categorical voting、ambiguity flag generation。
- 明确禁用：camera-only 2D 无可靠 3D 时直接入库。
- 每阶段保留 provenance/fusion lineage 追踪。

## Acceptance criteria
- `pytest -q tests/test_ica_generator.py` 通过。
- 同一输入多次运行结果字节级一致（determinism test）。

## Suggested branch name
- `task/02-ica-generator`

## Whether GPU is required
No.

## Whether real data is required
No.
