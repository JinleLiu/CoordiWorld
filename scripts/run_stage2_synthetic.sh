#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[stage2 synthetic] validating pairwise ranking batch schema"
echo "[stage2 synthetic] dry-run only: no ranking model training is executed"

python - <<'PY'
from __future__ import annotations

import json

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Stage II synthetic dry-run requires the optional train dependency. "
        "Install with: pip install -e '.[dev,train]'"
    ) from exc

from coordiworld.training.pairwise_schema import (
    PAIRWISE_RANKING_BATCH_FIELDS,
    PairwiseRankingBatch,
    validate_pairwise_ranking_batch,
)

batch = PairwiseRankingBatch(
    ego_history=torch.zeros(2, 3, 7),
    agent_history=torch.zeros(2, 3, 32, 16),
    map_history=torch.zeros(2, 3, 24, 10),
    action_tokens=torch.zeros(2, 4, 5, 6),
    agent_mask=torch.ones(2, 3, 32),
    map_mask=torch.ones(2, 3, 24),
    candidate_mask=torch.ones(2, 4),
    candidate_scores=torch.tensor([[0.10, 0.35, 0.60, 0.80], [0.20, 0.25, 0.55, 0.90]]),
    preferred_indices=torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.long),
    dispreferred_indices=torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long),
    pairwise_margins=torch.tensor([[0.25, 0.50, 0.20], [0.05, 0.30, 0.35]]),
    candidate_metadata=[
        {"candidate_pool_type": "synthetic_shared", "real_data": False},
        {"candidate_pool_type": "synthetic_shared", "real_data": False},
    ],
)
validate_pairwise_ranking_batch(batch)

print(
    json.dumps(
        {
            "dry_run": True,
            "real_data": False,
            "benchmark_result": False,
            "stage": "stage2_pairwise_schema_synthetic_dry_run",
            "field_contract": list(PAIRWISE_RANKING_BATCH_FIELDS),
            "batch_size": batch.batch_size,
            "candidate_count": batch.candidate_count,
            "horizon_steps": batch.horizon_steps,
            "pair_count": batch.pair_count,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
