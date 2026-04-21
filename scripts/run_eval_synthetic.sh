#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "[eval synthetic] running synthetic evaluator diagnostics"
echo "[eval synthetic] dry-run only: no real NAVSIM/OpenScene result is produced"

python - <<'PY'
from __future__ import annotations

import json

from coordiworld.evaluation.calibration_metrics import compute_calibration_metrics
from coordiworld.evaluation.navsim_metrics import NAVSIMMetricAdapter
from coordiworld.evaluation.ranking_metrics import compute_ranking_metrics
from coordiworld.evaluation.robustness import compute_ranking_stability

predicted_j = [0.12, 0.42, 0.28, 0.75]
target_risk = [0.10, 0.55, 0.25, 0.90]
collision_labels = [0, 0, 0, 1]
violation_labels = [0, 1, 0, 1]
ranking = compute_ranking_metrics(
    predicted_j,
    target_risk,
    collision_labels,
    violation_labels,
)
calibration = compute_calibration_metrics([0.05, 0.30, 0.70, 0.95], [0, 0, 1, 1], n_bins=4)
stability = compute_ranking_stability(predicted_j, [0.14, 0.39, 0.32, 0.72])
navsim_stub = NAVSIMMetricAdapter(dry_run=True).evaluate([])

print(
    json.dumps(
        {
            "dry_run": True,
            "real_data": False,
            "benchmark_result": False,
            "note": "Synthetic smoke diagnostics only; do not report as paper metrics.",
            "ranking_metrics": {
                "spearman": ranking.spearman,
                "kendall": ranking.kendall,
                "ndcg_at_3": ranking.ndcg_at_3,
                "top1_collision": ranking.top1_collision,
                "top1_violation": ranking.top1_violation,
                "selected_index": ranking.selected_index,
            },
            "calibration_metrics": {
                "ece": calibration.ece,
                "brier_score": calibration.brier_score,
            },
            "robustness_metrics": {
                "top1_unchanged": stability.top1_unchanged,
                "kendall": stability.kendall,
                "mean_abs_score_delta": stability.mean_abs_score_delta,
            },
            "navsim_metrics": {
                "dry_run": navsim_stub.dry_run,
                "metrics": navsim_stub.metrics,
                "message": navsim_stub.message,
            },
        },
        indent=2,
        sort_keys=True,
    )
)
PY
