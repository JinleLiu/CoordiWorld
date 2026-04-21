#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[stage1 synthetic] running CPU one-batch smoke training"
echo "[stage1 synthetic] no real dataset, checkpoint, or benchmark metric is used"

python - <<'PY'
from __future__ import annotations

import json

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Stage I synthetic smoke requires the optional train dependency. "
        "Install with: pip install -e '.[dev,train]'"
    ) from exc

from coordiworld.data.candidate_pool import CandidatePoolConfig
from coordiworld.data.synthetic import SyntheticDatasetConfig, SyntheticScenarioDataset
from coordiworld.models.coordiworld import CoordiWorldModel
from coordiworld.training.stage1_rollout import build_stage1_batch
from coordiworld.training.trainer import Stage1RolloutTrainer

torch.manual_seed(17)

dataset = SyntheticScenarioDataset(
    SyntheticDatasetConfig(
        num_samples=2,
        history_length=2,
        candidate_pool_config=CandidatePoolConfig(horizon_steps=3),
    )
)
samples = [dataset[0], dataset[1]]
batch = build_stage1_batch(samples, device="cpu")
model = CoordiWorldModel(hidden_dim=32, num_heads=4, num_layers=1)
trainer = Stage1RolloutTrainer(model, device="cpu")
result = trainer.train_one_batch(batch)

print(
    json.dumps(
        {
            "dry_run": True,
            "real_data": False,
            "benchmark_result": False,
            "stage": "stage1_rollout_synthetic_smoke",
            "batch_size": len(samples),
            "history_length": int(batch.ego_history.shape[1]),
            "candidate_count": int(batch.action_tokens.shape[1]),
            "horizon_steps": int(batch.action_tokens.shape[2]),
            "loss_value": result.loss_value,
            "loss_components": result.components,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
