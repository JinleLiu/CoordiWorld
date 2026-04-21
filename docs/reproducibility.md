# Reproducibility

本文档说明 CoordiWorld 当前实验 pipeline 的可复现边界。现阶段只提供 synthetic smoke 与 dry-run plumbing，不生成真实论文结果，不硬编码 IOTJ-V3 表格数字。

## 环境安装

```bash
git clone <repo-url> CoordiWorld
cd CoordiWorld
conda create -n coordiworld python=3.10 -y
conda activate coordiworld
pip install -e ".[dev,train]"
```

如果只运行纯 evaluation dry-run，`.[dev]` 通常足够；Stage I/Stage II synthetic 脚本会导入 PyTorch，因此需要 `train` optional dependency。

## 数据路径

真实数据路径必须通过环境变量或私有 `.env` 设置，不要提交到 git：

```bash
export DATA_ROOT=/path/to/datasets
export NAVSIM_ROOT=/path/to/datasets/navsim
export OPENSCE_ROOT=/path/to/datasets/opensce
export NUSCENES_ROOT=/path/to/datasets/nuscenes
export WAYMO_ROOT=/path/to/datasets/waymo
export OUTPUT_ROOT=/path/to/coordiworld/outputs
export CHECKPOINT_ROOT=/path/to/coordiworld/checkpoints
export WANDB_MODE=offline
```

仓库中的脚本不会扫描 `data/`、`outputs/`、`checkpoints/` 或 `wandb/`。真实结果、模型权重和日志产物不得提交。

## Synthetic Smoke Test

无真实数据环境下可以运行：

```bash
bash scripts/run_stage1_synthetic.sh
bash scripts/run_stage2_synthetic.sh
bash scripts/run_eval_synthetic.sh
python scripts/make_tables_from_results.py --dry-run
```

这些命令只验证接口、schema 和 metric plumbing。输出如果带有 `dry_run: true` 或 `benchmark_result: false`，只能作为工程 smoke 记录，不能写成 benchmark 结论。

## NAVSIM/OpenScene 接入位置

- NAVSIM 数据接口：`src/coordiworld/data/navsim_adapter.py`
- SceneSummary generator：`src/coordiworld/scene_summary/generator.py`
- shared candidate pool：`src/coordiworld/data/candidate_pool.py`
- NAVSIM metric wrapper stub：`src/coordiworld/evaluation/navsim_metrics.py`
- 配置模板：`experiments/navsim_v2_shared_candidates.yaml`

接入真实 NAVSIM/OpenScene 时，必须使用官方数据和官方 metric wrapper。官方 wrapper 不可用时，只能返回 dry-run stub 或抛出清晰错误，不得伪造 EPDMS。

## 结果表格

`scripts/make_tables_from_results.py` 只接受真实结果 JSON/CSV 输入来生成结果表。没有真实结果文件时，默认报错；如果显式传入 `--dry-run`，它只生成带有 `DRY-RUN PLACEHOLDER - NOT A BENCHMARK RESULT` 的占位表。

禁止事项：

- 不要复制、硬编码或声称复现 IOTJ-V3 表格数字。
- 不要把 synthetic smoke 指标当作真实 NAVSIM/EPDMS。
- 不要提交 `outputs/`、`checkpoints/`、`wandb/`、真实数据或模型权重。
