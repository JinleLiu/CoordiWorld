# COORDIWORLD_SPEC

## 1. Problem setup
CoordiWorld is defined as a **fixed candidate-set trajectory evaluator** (not a generator).

- Input A: SceneSummary history `S_{t-h:t}`.
- Input B: candidate ego trajectory set `T_t = {tau^(m)}`.
- Output: selected trajectory `tau* = argmax_{tau in T_t} J(tau)` under calibrated evaluator scoring.

## 2. Structured scene interface
The evaluator consumes a structured scene tensor/object view with explicit token groups:

- `ego_token`: ego historical state, kinematics, footprint, and control-aligned context.
- `agent_tokens`: dynamic entities with geometry, motion, confidence, provenance, and ambiguity metadata.
- `map_tokens`: lane centerline/topology, boundary, stopline, crosswalk, drivable area, traffic-rule anchors.
- `confidence/provenance cues`: per-field uncertainty, modality/source IDs, fusion lineage, and conflict flags.

All interfaces MUST preserve deterministic field semantics and auditable source trace.

## 3. Candidate-conditioned rollout
For each candidate `tau`:

1. **Future ego motion is fixed by candidate**: ego future trajectory is directly conditioned by `tau`.
2. **Stochastic component is tracked-agent future evolution**: non-ego futures are predicted conditionally on scene context + candidate ego behavior.
3. Rollout head outputs per tracked entity and horizon step:
   - residual motion (`Δx`, `Δy`, `Δyaw`, optional velocity residuals),
   - existence probability,
   - position covariance (at least planar covariance in ego-aligned coordinates).

## 4. Risk decomposition and final score
Risk heads decompose evaluator outputs into:

- collision risk,
- rule-violation risk (map-grounded),
- predictive uncertainty,
- post-hoc calibration mapping.

Final score function:

`J(tau) = Calibrate(Combine(R_collision(tau), R_rule(tau), U_pred(tau), optional comfort/efficiency priors))`.

Implementation must keep decomposition inspectable for audit and ablation.

## 5. Entity attribution
Attribution is post-hoc and entity-centric:

- Counterfactual entity masking/removal over structured inputs.
- Measure score/risk delta to identify influential entities.
- Attribution output is **audit-only** and MUST NOT be directly used as ranking signal in core ordering logic.

## 6. Training protocol
- **Stage I**: structured rollout pretraining (next-state/residual/existence/covariance objectives).
- **Stage II**: pairwise ranking fine-tuning over candidates in shared-candidate protocol.
- **Post-hoc calibration**: fit calibrator on held-out split for score-to-risk alignment.

When real benchmark wrappers/data are unavailable, only interfaces, stubs, dry-run and synthetic tests are allowed.
