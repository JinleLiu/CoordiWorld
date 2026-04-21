# TASK finalize GitHub project

## 日期
2026-04-21

## 本轮目标
将当前仓库整理为清晰、可复现、可上传 GitHub 的 Python research project。重点是 README、数据接口、CLI、配置、examples、CI/tests/lint 和文档，不重新设计论文方法，不扩展新算法。

## 初始状态
- 初始分支：`exp/pipeline-and-docs`
- 工作区：干净
- 已创建工作分支：`project-finalize-github`
- 远程：`origin https://github.com/JinleLiu/CoordiWorld.git`
- Python：`3.12.4`
- pytest：`9.0.3`

## 范围控制
- 不扫描或处理 `data/`、`outputs/`、`checkpoints/`、`wandb/`。
- 不修改 `.env` 或 `.idea/`。
- 不伪造 NAVSIM/OpenScene/nuScenes/Waymo/Bench2Drive 结果。
- 不硬编码论文表格数值。
- 不提交真实数据、模型权重、日志或服务器私有路径。

## 当前进展
- 已补项目审计文档 `docs/PROJECT_AUDIT.md`。
- 已补统一数据错误类型、adapter protocol、registry、JSONL adapter 和 native adapter boundaries。
- 已补 dataset configs 与 minimal examples。
- 已补 `validate_data` CLI 和 synthetic/jsonl smoke path。

## 待验证
- targeted pytest。
- synthetic/jsonl CLI smoke。
- CLI help。
- `ruff check src tests`。
