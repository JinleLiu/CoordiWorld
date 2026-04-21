# Troubleshooting

## PyCharm remote conda 卡住

- 确认远程解释器指向已创建的 conda env。
- 在服务器 shell 中先运行：

```bash
conda activate coordiworld
python -V
python -m pytest --version
```

- 如果 PyCharm indexing 卡住，先用终端执行 `pip install -e ".[dev]"`，再重新选择 interpreter。

## `ModuleNotFoundError: coordiworld.data`

常见原因是没有以 editable 模式安装：

```bash
pip install -e ".[dev]"
```

也可以临时设置：

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

## `.gitignore` 误伤 `src/coordiworld/data/`

`.gitignore` 必须使用根目录锚定：

```gitignore
/data/
```

不要写成：

```gitignore
data/
```

否则可能影响 `src/coordiworld/data/` 的跟踪状态。

## `MissingDependencyError`

这表示真实数据 adapter 需要官方依赖，例如 NAVSIM、OpenScene、nuScenes devkit 或 Waymo Open Dataset。当前仓库不会附带这些依赖，也不会伪造样本。

解决方式：

1. 安装官方数据 API/devkit。
2. 设置对应 root 环境变量。
3. 再运行 `python -m coordiworld.cli.validate_data ...`。

## `DATA_ROOT` 或 dataset root 未设置

真实数据路径必须来自环境变量或私有配置：

```bash
export DATA_ROOT=/path/to/datasets
export NAVSIM_ROOT=/path/to/datasets/navsim
export OPENSCE_ROOT=/path/to/datasets/opensce
export NUSCENES_ROOT=/path/to/datasets/nuscenes
export WAYMO_ROOT=/path/to/datasets/waymo
```

不要把真实服务器路径提交到 git。

## JSONL adapter 报 `DatasetFormatError`

检查每行是否包含：

- `scene_id`
- `timestamp`
- `scene_summary_history`
- `candidate_trajectories`
- `logged_ego_future`
- `labels`

可参考 `examples/data/scenario_sample_minimal.jsonl`。
