# Examples

本目录保存可提交到 GitHub 的最小示例数据，不包含真实 benchmark 数据。

- `data/scene_summary_minimal.json`：一个结构化 `SceneSummary` 示例。
- `data/scenario_sample_minimal.jsonl`：标准化 JSONL scenario sample 示例，每行一个 `BaseScenarioSample`。

可运行检查：

```bash
python -m coordiworld.cli.validate_data --dataset jsonl --config configs/datasets/jsonl_example.yaml
```

这些示例只用于 smoke test，不能作为 NAVSIM/OpenScene/nuScenes/Waymo 结果。
