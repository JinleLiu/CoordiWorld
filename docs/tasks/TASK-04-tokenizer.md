# TASK-04: tokenizer

## Goal
实现 structured scene tokenizer，将 SceneSummary 转成 ego/agent/map token 序列与 mask。

## Allowed files
- `src/coordiworld/model/tokenizer.py`
- `tests/test_tokenizer.py`

## Read first
- `docs/specs/COORDIWORLD_SPEC.md`
- `docs/specs/SCENESUMMARY_SPEC.md`

## Out of scope
- rollout 网络
- 训练流程

## Implementation details
- 明确 token group 边界与字段映射。
- 保留 confidence/provenance/ambiguity 的编码通道。
- 支持 deterministic padding/truncation 规则。

## Acceptance criteria
- `pytest -q tests/test_tokenizer.py` 通过。
- synthetic 样本下 token shape 与 mask 行为可复现。

## Suggested branch name
- `task/04-tokenizer`

## Whether GPU is required
No.

## Whether real data is required
No.
