# TASK-00 Paper-to-Code Dev Log

## Date
2026-04-21

## Scope
将 CoordiWorld 与 InfoCoordiBridge 论文要点沉淀为工程规格文档（specs）与分阶段任务文件（tasks），不实现 `src/` 代码。

## Inputs reviewed
- `AGENTS.md`
- `README.md`
- `docs/project_brief.md`
- `papers/IOTJ-V3.pdf`（环境缺少 PDF 解析工具，仅按仓库已有锚点与任务要求抽取）
- `papers/manuscript-InfoCoordiBridge.pdf`（同上）

## Output artifacts
- specs:
  - `docs/specs/COORDIWORLD_SPEC.md`
  - `docs/specs/SCENESUMMARY_SPEC.md`
  - `docs/specs/EXPERIMENT_PROTOCOL.md`
  - `docs/specs/DATA_ADAPTER_SPEC.md`
  - `docs/specs/METRICS_SPEC.md`
- tasks:
  - `docs/tasks/TASK-00-repo-scaffold.md`
  - `docs/tasks/TASK-01-scene-summary-schema.md`
  - `docs/tasks/TASK-02-ica-generator.md`
  - `docs/tasks/TASK-03-data-adapter-candidate-pool.md`
  - `docs/tasks/TASK-04-tokenizer.md`
  - `docs/tasks/TASK-05-rollout-model.md`
  - `docs/tasks/TASK-06-risk-evaluator.md`
  - `docs/tasks/TASK-07-stage1-training.md`
  - `docs/tasks/TASK-08-stage2-ranking.md`
  - `docs/tasks/TASK-09-calibration-evaluation.md`
  - `docs/tasks/TASK-10-attribution-robustness.md`

## Notes
- 遵循“fixed candidate-set evaluator, not generator”主约束。
- 明确无真实数据/无官方 wrapper 时仅允许接口、dry-run、synthetic tests。
- 未创建任何 `src/`、训练脚本或实验结果表格。
