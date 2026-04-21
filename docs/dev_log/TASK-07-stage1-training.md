# TASK-07 stage1 training

## 日期
2026-04-21

## 本轮目标
实现 Stage I structured rollout pretraining 的 loss 和 synthetic one-batch smoke training，不接入真实 NAVSIM，不生成论文指标。

## 完成内容
1. 新增 `src/coordiworld/training/losses.py`，实现 xy Gaussian NLL、yaw/velocity Huber、existence BCE、risk BCE 和加权 Stage I loss。
2. 新增 `src/coordiworld/training/stage1_rollout.py`，将 synthetic `BaseScenarioSample` 构造成 Stage I batch，并使用 logged ego future 作为单候选 action token。
3. 新增 `src/coordiworld/training/trainer.py`，提供 one-batch synthetic smoke train step 和 non-mutating loss evaluation。
4. 新增 `tests/test_stage1_losses.py` 与 `tests/test_stage1_smoke_train.py`，覆盖 loss 数值行为、反传、batch shape 和 one-batch 参数更新。

## 范围控制
- 未实现完整训练循环。
- 未接入真实 NAVSIM。
- 未读取真实数据。
- 未生成论文指标或 benchmark 结论。
- 未要求 GPU。

## 验收命令
```bash
pytest tests/test_stage1_losses.py tests/test_stage1_smoke_train.py -q
```

## 已知限制
- trainer 仅支持 one-batch smoke training，不包含 dataloader、checkpoint 或 scheduler。
- risk BCE 使用 rollout 输出的 existence logits 聚合作为 synthetic auxiliary logit，不代表最终风险评估模型。
- Stage I target 来自 synthetic fixture 的 future agent labels，仅用于接口和梯度 smoke test。
