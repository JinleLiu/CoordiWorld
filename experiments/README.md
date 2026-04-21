# CoordiWorld Experiments

本目录只保存实验配置模板和 synthetic smoke 配置，不保存真实结果、`outputs/`、`checkpoints/` 或 `wandb/` 产物。

## 配置文件

- `synthetic_smoke.yaml`：无真实数据环境下的 pipeline smoke 配置。它只用于验证 Stage I、Stage II schema 和 evaluation plumbing。
- `navsim_v2_shared_candidates.yaml`：NAVSIM/OpenScene 接入模板。真实路径必须来自环境变量，官方 metric wrapper 不可用时只能 dry-run，不得生成或声明真实 EPDMS。

## 本地 dry-run

```bash
bash scripts/run_stage1_synthetic.sh
bash scripts/run_stage2_synthetic.sh
bash scripts/run_eval_synthetic.sh
python scripts/make_tables_from_results.py --dry-run
```

这些命令不会读取真实数据，也不会写入模型权重。输出中的 `dry_run: true` 和 `benchmark_result: false` 表示它们不能作为论文或 benchmark 指标引用。
