# EXPERIMENT_PROTOCOL

## 1. Shared-candidate evaluator protocol
All evaluator comparisons must use a **shared candidate pool per scene/time step**.
Every compared method receives exactly the same candidate set and ranks/selects within that set.

## 2. Candidate pool construction
Candidate pool should include:

- nominal proposal,
- speed-scaled variants,
- lateral-shift variants,
- curvature-perturbed variants.

Generation/config should be deterministic under seed control and recorded in metadata.

## 3. Primary metrics (protocol-level)
- EPDMS
- NC
- DAC
- DDC
- TLC
- EP
- TTC
- LK
- HC
- EC

## 4. Evaluator-specific ranking/safety metrics
- Spearman
- Kendall
- NDCG@3
- top-1 collision
- top-1 violation

## 5. Calibration metrics
- ECE
- Brier Score

## 6. Robustness tests
- confidence noise
- provenance masking
- evidence dropout

## 7. Auditability metrics
- RiskDrop@K
- EntityRecall@K

## 8. Data/wrapper availability declaration
If official metric wrappers or real datasets are unavailable, implementation scope is limited to:

- interface contracts,
- dry-run pipelines,
- synthetic tests.

No real benchmark conclusion should be generated or claimed in that mode.
