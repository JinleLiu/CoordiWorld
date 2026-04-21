# AGENTS.md

## Communication
1. Default to Chinese for all explanations and progress updates.
2. Keep Python package names, class names, function names, field names, config keys, and file names in English.

## Workflow Discipline
3. Complete exactly one clearly scoped task per turn; do not silently bundle unrelated work.
4. Do not scan or process `data/`, `outputs/`, `checkpoints/`, or `wandb/` unless the user explicitly overrides this rule.
5. Do not commit real datasets, model weights, or large log artifacts.
6. Do not hardcode or claim paper table metrics as reproduced results.
7. Every new module must include `pytest` tests.
8. When real datasets are unavailable, use synthetic fixtures, mock tests, and clearly marked stubs.
9. Do not run `git commit` automatically unless the user explicitly asks.
10. Before modifying files outside the requested scope, explain why and wait for confirmation.
11. SceneSummary-related logic must be deterministic and must not call any LLM service.

## Output Requirements (must include at task completion)
12. Modified file list.
13. Design choices.
14. Suggested test commands.
15. Known limitations.
16. Next-step suggestions.

## Project Guardrails for This Repository
- CoordiWorld is a fixed candidate-set trajectory evaluator, not a trajectory generator.
- Inputs should align with paper anchors: SceneSummary history `S_{t-h:t}` and candidate trajectories `T_t = {tau^(m)}`.
- Evaluator outputs should include collision risk, map-grounded rule-violation risk, predictive uncertainty, calibrated score `J(tau)`, and post-hoc entity-level attribution.
- SceneSummary generation should follow InfoCoordiBridge/ICA principles: coordinate normalization, cross-source entity alignment, conflict-aware fusion, and provenance/ambiguity flags.
- Without NAVSIM/OpenScene/NuScenes/Waymo data, only implement interfaces, mocks, synthetic smoke tests, and clearly marked stubs.
