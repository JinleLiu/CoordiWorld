# TASK-00 local dev env

## 日期
2026-04-21

## 本轮目标
补充服务器与 PyCharm 远程开发环境说明，并提供只读环境变量检查入口。

## 完成内容
1. 新增 `.env.example`，列出数据、输出、checkpoint 与 `WANDB_MODE` 的必要环境变量。
2. 新增 `docs/setup_server_pycharm.md`，说明服务器 clone、conda env、`pip install -e ".[dev]"`、PyCharm remote interpreter 与 pytest smoke test。
3. 新增 `scripts/check_env.py`，只检查环境变量是否设置、路径是否存在且可访问。
4. 新增 `tests/test_env_config.py`，校验 `.env.example` 与检查脚本声明的必要字段一致。

## 范围控制
- 未实现任何核心算法。
- 未读取真实数据目录内容。
- 未下载数据。
- 未调用 GPU。
- 未提交真实数据路径、权重或日志产物。

## 验收命令
```bash
pytest tests/test_env_config.py -q
python scripts/check_env.py --help
```

## 已知限制
- `.env.example` 仅是模板，真实服务器路径必须由开发者在本地或 PyCharm 配置中填写。
- 当前脚本不会创建 `OUTPUT_ROOT` 或 `CHECKPOINT_ROOT`；如目录不存在会报告失败。
