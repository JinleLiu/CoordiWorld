# TASK-05 rollout model

## 日期
2026-04-21

## 本轮目标
实现 CoordiWorld structured rollout model 的最小可训练 PyTorch forward，不实现完整训练循环、评估表格或真实数据适配。

## 完成内容
1. 新增 `src/coordiworld/models/encoder.py`，实现基于 `nn.TransformerEncoder` 的 scene history temporal encoder。
2. 新增 `src/coordiworld/models/heads.py`，实现 action-conditioned per-agent residual/existence/covariance output head。
3. 新增 `src/coordiworld/models/rollout.py`，实现 residual composition 和 `[B,M,H,N,...]` structured rollout。
4. 新增 `src/coordiworld/models/coordiworld.py`，提供最小 `CoordiWorldModel` wrapper。
5. 新增 `tests/test_rollout_model.py`，覆盖 CPU forward shape、action conditioning、yaw residual composition、mask 行为和 backward smoke test。

## 范围控制
- 未实现完整训练循环。
- 未实现评估表格。
- 未实现真实数据适配。
- 未实现风险评估、ranking fine-tuning 或 calibration。
- 未读取真实数据。
- 未调用 GPU。

## 验收命令
```bash
pytest tests/test_rollout_model.py -q
```

## 已知限制
- 当前模型是最小可训练 forward，未定义 loss recipe 或 optimizer step。
- Transformer encoder 使用 token mean pooling 汇总 agent/map token groups，后续可扩展为 entity-level attention。
- covariance 输出为正 `sigma` 与对应 `log_variance`，暂未输出完整 2x2 covariance matrix。
