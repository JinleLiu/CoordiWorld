# Project Structure

```text
CoordiWorld/
  README.md
  AGENTS.md
  pyproject.toml
  .env.example
  configs/
  docs/
  examples/
  experiments/
  scripts/
  src/coordiworld/
  tests/
```

## `src/coordiworld`

- `scene_summary/`: structured `SceneSummary` schema, JSON I/O, validation, ICA-style transforms/association/fusion/generation.
- `data/`: sample contracts, registry, synthetic dataset, JSONL adapter, native dataset adapter boundaries, candidate pool, collate helpers.
- `tokens/`: `SceneSummary`, map token, and candidate trajectory tokenization.
- `models/`: minimal PyTorch structured rollout model.
- `risks/`: collision, rule violation, uncertainty, calibration, and final score `J`.
- `evaluation/`: ranking/calibration metrics, NAVSIM dry-run adapter, robustness, auditability.
- `attribution/`: post-hoc entity masking and counterfactual attribution.
- `training/`: Stage I losses/smoke trainer and Stage II pairwise ranking batch schema.
- `cli/`: command entry points for validation, dry-runs, and placeholders.
- `visualization/`: reserved package namespace.

## `configs`

- `configs/datasets/`: dataset adapter templates.
- `configs/model/`: model config templates.
- `configs/default.yaml`: scaffold default config.

## `examples`

Minimal `SceneSummary` and standardized JSONL examples for smoke tests. These are not benchmark data.

## `experiments`

Synthetic smoke and real-data template configs. Do not store real results here unless they are small, audited, and explicitly intended for git.

## `scripts`

Local smoke scripts and table-generation helper. They must not write real datasets, checkpoints, or fake benchmark results.

## `tests`

Synthetic/unit tests only. Tests must not require real data, GPU, downloads, or official benchmark wrappers.
