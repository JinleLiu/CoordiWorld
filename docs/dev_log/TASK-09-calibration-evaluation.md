# TASK-09 calibration evaluation

## 日期
2026-04-21

## 本轮目标
实现 evaluator ranking metrics、calibration metrics 与 NAVSIM metric wrapper stub，不生成真实实验表格，不伪造 EPDMS。

## 完成内容
1. 新增 `src/coordiworld/evaluation/ranking_metrics.py`，实现 Spearman、Kendall、NDCG@3、top-1 collision、top-1 violation。
2. 新增 `src/coordiworld/evaluation/calibration_metrics.py`，实现 ECE、Brier Score 与 reliability-bin diagnostics。
3. 新增 `src/coordiworld/evaluation/navsim_metrics.py`，定义 `NAVSIMMetricAdapter` 接口和明确标注的 dry-run stub。
4. 新增 `tests/test_ranking_metrics.py` 与 `tests/test_calibration_metrics.py`，使用 synthetic arrays 覆盖指标行为与 NAVSIM stub。

## 范围控制
- 未生成真实实验表格。
- 未伪造 EPDMS。
- 未读取真实 NAVSIM 数据。
- 未接入官方 NAVSIM wrapper。
- 未调用 GPU。

## 验收命令
```bash
pytest tests/test_ranking_metrics.py tests/test_calibration_metrics.py -q
```

## 已知限制
- ranking metrics 默认按 risk score lower-is-better 解释。
- calibration metrics 只覆盖 binary probability calibration。
- `NAVSIMMetricAdapter` 在没有官方 wrapper 时只会抛出清晰错误或返回 dry-run `None` metrics。
