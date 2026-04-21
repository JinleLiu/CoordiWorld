# METRICS_SPEC

## 1. Ranking metrics
Ranking quality over shared candidate sets should include:

- Spearman correlation,
- Kendall rank correlation,
- NDCG@3,
- top-1 consistency checks.

## 2. Calibration metrics
Calibration quality should include:

- ECE,
- Brier Score,
- optional reliability-bin summaries for diagnostics.

## 3. Risk metrics
Risk evaluation should include task-aligned risk families:

- collision-related risk metrics,
- map rule-violation risk metrics,
- uncertainty-aware risk summaries.

## 4. Robustness metrics
Robustness under perturbation should include:

- confidence noise sensitivity,
- provenance masking sensitivity,
- evidence dropout sensitivity.

## 5. Attribution metrics
Auditability and attribution quality should include:

- RiskDrop@K,
- EntityRecall@K,
- counterfactual consistency checks.

## 6. Stub/adapter strategy without official NAVSIM wrapper
When real NAVSIM metric wrappers are absent:

- define metric interfaces and adapter abstraction first,
- provide deterministic stub implementations for dry-run,
- validate plumbing using synthetic fixtures,
- clearly mark outputs as non-benchmark.
