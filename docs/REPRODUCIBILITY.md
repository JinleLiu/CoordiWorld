# Reproducibility

CoordiWorld 当前提供 synthetic smoke test、标准化 JSONL 接口和真实数据 adapter 边界。没有官方数据/API/wrapper 时，不生成真实 benchmark 结论。

## Synthetic Baseline

先在无真实数据环境中确认工程链路：

```bash
python -m coordiworld.cli.validate_data --dataset synthetic --max-samples 2
python -m coordiworld.cli.validate_data --dataset jsonl --config configs/datasets/jsonl_example.yaml --max-samples 2
bash scripts/run_eval_synthetic.sh
python -m pytest -q
ruff check src tests
```

这些结果只表示接口和 smoke test 正常，不是论文结果。

## Transition to Real Data

1. 安装官方 dataset/devkit/wrapper。
2. 设置 `NAVSIM_ROOT`、`OPENSCE_ROOT`、`NUSCENES_ROOT` 或 `WAYMO_ROOT`。
3. 使用 native adapter 或先转换为标准化 JSONL。
4. 运行 `validate_data`，确认 `BaseScenarioSample` 和 `SceneSummary` schema 通过。
5. 使用 shared candidate pool，确保比较方法收到同一候选集合。
6. 用官方 metric wrapper 生成真实 benchmark 指标。
7. 只从真实、可审计的 JSON/CSV result 文件生成表格。

## Forbidden Claims

- 不声称已复现 NAVSIM EPDMS，除非官方 wrapper 和真实结果可审计。
- 不把 synthetic smoke 指标写成真实 benchmark。
- 不硬编码 IOTJ-V3 或任何论文表格数字。
- 不提交真实数据、模型权重、日志、API key 或服务器私有路径。

## Result Table Rule

`scripts/make_tables_from_results.py` 默认要求 `--input` 指向真实 JSON/CSV 结果。没有输入时会报错；只有显式 `--dry-run` 才会生成 clearly marked placeholder table。
