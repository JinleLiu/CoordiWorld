# TASK-11 experiment pipeline

## 日期
2026-04-21

## 本轮目标
创建实验 pipeline 的 dry-run 和 synthetic smoke test 脚本，补充实验配置与复现说明，不生成真实论文结果，不硬编码 IOTJ-V3 表格数字。

## 完成内容
1. 新增 `scripts/run_stage1_synthetic.sh`，在 CPU 上运行 Stage I one-batch synthetic smoke training。
2. 新增 `scripts/run_stage2_synthetic.sh`，构造并验证 synthetic `PairwiseRankingBatch`，不执行真实 Stage II 训练。
3. 新增 `scripts/run_eval_synthetic.sh`，计算 synthetic ranking/calibration/robustness diagnostics，并返回 NAVSIM dry-run stub。
4. 新增 `scripts/make_tables_from_results.py`，从真实 JSON/CSV 结果生成 Markdown/CSV 表；无真实输入时默认报错，`--dry-run` 只生成明确标记的占位表。
5. 新增 `experiments/README.md`、`experiments/synthetic_smoke.yaml` 与 `experiments/navsim_v2_shared_candidates.yaml`。
6. 新增 `docs/reproducibility.md`，说明环境安装、数据路径、synthetic smoke、NAVSIM/OpenScene 接入位置和禁止伪造指标规则。

## 范围控制
- 未读取真实数据。
- 未写入 `outputs/`、`checkpoints/` 或 `wandb/`。
- 未生成真实论文表格。
- 未伪造 EPDMS 或 NAVSIM 指标。
- 未改变模型、风险分数或 evaluator 的定义。

## 验收命令
```bash
bash scripts/run_eval_synthetic.sh
python scripts/make_tables_from_results.py --help
```

## 已知限制
- Stage II 脚本当前只验证 pairwise ranking batch schema，因为 ranking trainer 尚未实现。
- `make_tables_from_results.py` 只做轻量表格汇总，不负责统计显著性或多 seed 聚合。
- NAVSIM 配置文件是接入模板，真实 benchmark 仍依赖官方数据与官方 metric wrapper。
