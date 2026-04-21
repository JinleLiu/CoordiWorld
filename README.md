# CoordiWorld

CoordiWorld is a provenance-aware structured world model for auditable fixed candidate-set trajectory evaluation in autonomous driving.

CoordiWorld is not a trajectory generator. It evaluates a fixed candidate set of ego trajectories using structured world state, predicted interactions, rule grounding, uncertainty, calibration, and post-hoc attribution.

## What This Repository Contains

| Area | Status | Notes |
| --- | --- | --- |
| `SceneSummary` schema | Implemented | Structured world state dataclasses, JSON I/O, validator. |
| ICA-style `SceneSummary` generator | Implemented for synthetic/unit cases | Coordinate normalization, association, conflict-aware fusion, provenance flags. |
| data adapters | Synthetic/JSONL runnable; native adapters are strict stubs | NAVSIM/OpenScene/nuScenes/Waymo require official data and dependencies. |
| shared candidate pool | Implemented | Nominal, speed-scaled, lateral-shift, curvature-perturbed variants. |
| tokenizer | Implemented | Scene, map, and action/candidate trajectory tokenizers. |
| structured rollout model | Minimal PyTorch implementation | CPU smoke tests only; not an official reproduction. |
| risk evaluator | Implemented | Collision, rule violation, uncertainty, calibration, final score `J(tau)`. |
| evaluation metrics | Implemented | Ranking, calibration, robustness, auditability metrics. |
| attribution utilities | Implemented | Post-hoc entity masking/counterfactual attribution. |
| CLI and synthetic smoke tests | Implemented |


## Method Overview

CoordiWorld consumes:

- `SceneSummary` history `S_{t-h:t}`: structured ego, agents, map tokens, provenance, and metadata.
- fixed candidate ego trajectories `T_t = {tau^(m)}` shared across methods.

For each candidate trajectory, the pipeline supports:

1. candidate-conditioned structured rollout,
2. collision-risk estimation,
3. map-grounded rule-violation risk,
4. predictive uncertainty,
5. calibrated lower-is-better score `J(tau)`,
6. post-hoc entity-level attribution for auditability.

`SceneSummary` is a structured world-state interface, not a natural-language summary. The generation utilities follow InfoCoordiBridge/ICA principles: coordinate normalization, cross-source entity alignment, conflict-aware attribute fusion, and provenance-aware structured summaries.

## Repository Structure

```text
src/coordiworld/
  scene_summary/   SceneSummary schema, I/O, validation, ICA-style generation helpers
  data/            Base sample contracts, registry, synthetic/jsonl/native adapters
  tokens/          Scene, map, and action tokenizers
  models/          Minimal structured rollout model
  risks/           Collision, rule violation, uncertainty, calibration, score J
  evaluation/      Ranking, calibration, NAVSIM stub, robustness, auditability metrics
  attribution/     Post-hoc entity masking and counterfactual attribution
  training/        Stage I losses/smoke trainer and Stage II pairwise batch schema
  cli/             Dataset validation and dry-run command entry points
scripts/           Synthetic smoke scripts and table-generation helper
configs/           Model and dataset config templates
examples/          Minimal JSON/JSONL examples for smoke tests
docs/              Setup, data interface, reproducibility, audit, troubleshooting
tests/             Synthetic and unit tests
```

## Installation

```bash
git clone <repo-url> CoordiWorld
cd CoordiWorld
conda create -n coordiworld python=3.10 -y
conda activate coordiworld
pip install -e ".[dev]"
```

Optional model/training dependencies:

```bash
pip install -e ".[dev,model,train]"
```

Development checks:

```bash
python -m pytest -q
ruff check src tests
```

## Quickstart: Synthetic Smoke Test

These commands run without real datasets:

```bash
python -m coordiworld.cli.validate_data --dataset synthetic --max-samples 2
python -m coordiworld.cli.validate_data --dataset jsonl --config configs/datasets/jsonl_example.yaml
python -m coordiworld.cli.build_scene_summary --dataset synthetic --max-samples 1
bash scripts/run_eval_synthetic.sh
python -m pytest -q
```

Synthetic outputs are engineering smoke diagnostics only. They are not benchmark results.

## Data Setup

Real data paths must be provided through environment variables or private local config. Do not commit real paths.

```bash
export DATA_ROOT=/path/to/datasets
export NAVSIM_ROOT=/path/to/datasets/navsim
export OPENSCE_ROOT=/path/to/datasets/opensce
export NUSCENES_ROOT=/path/to/datasets/nuscenes
export WAYMO_ROOT=/path/to/datasets/waymo
export OUTPUT_ROOT=/path/to/coordiworld/outputs
export CHECKPOINT_ROOT=/path/to/coordiworld/checkpoints
export WANDB_MODE=offline
```

`.env.example` is only a template. `.env` and `.env.*` are ignored.

## Dataset Adapters

### `synthetic`

- Purpose: deterministic smoke tests and unit tests.
- Config: `configs/datasets/synthetic.yaml`
- Environment variable: none.
- Validate:

```bash
python -m coordiworld.cli.validate_data --dataset synthetic --max-samples 2
```

### `jsonl`

- Purpose: standardized intermediate format after converting real datasets.
- Config: `configs/datasets/jsonl_example.yaml`
- Environment variable: none for the example.
- Validate:

```bash
python -m coordiworld.cli.validate_data --dataset jsonl --config configs/datasets/jsonl_example.yaml
```

### NAVSIM / OpenScene

- Purpose: future real benchmark integration.
- Configs: `configs/datasets/navsim.yaml`, `configs/datasets/openscene.yaml`
- Environment variables: `NAVSIM_ROOT`, `OPENSCE_ROOT`
- Validate root/dependency boundary:

```bash
python -m coordiworld.cli.validate_data --dataset navsim --config configs/datasets/navsim.yaml
python -m coordiworld.cli.validate_data --dataset openscene --config configs/datasets/openscene.yaml
```

### nuScenes

- Purpose: optional SceneSummary/ICA data source.
- Config: `configs/datasets/nuscenes.yaml`
- Environment variable: `NUSCENES_ROOT`
  
### Waymo

- Purpose: optional SceneSummary/ICA data source.
- Config: `configs/datasets/waymo.yaml`
- Environment variable: `WAYMO_ROOT`

## Standardized JSONL Schema

`examples/data/scenario_sample_minimal.jsonl` contains one scenario sample per line. Required fields:

- `scene_id`
- `timestamp`
- `scene_summary_history`
- `candidate_trajectories` with shape `[M,H,3]`
- `logged_ego_future` with shape `[H,3]`
- `labels`

Recommended fields:

- `sample_id`
- `coordinate_frame`
- `future_agents`
- `provenance`
- `quality_flags`
- `metadata`

Each `SceneSummary` record uses the schema in `src/coordiworld/scene_summary/schema.py`.

## Running Training / Evaluation

Synthetic smoke commands:

```bash
bash scripts/run_stage1_synthetic.sh
bash scripts/run_stage2_synthetic.sh
bash scripts/run_eval_synthetic.sh
```

Real-data expected workflow:

1. install official dataset dependencies,
2. configure dataset roots via environment variables,
3. convert or expose real samples as `BaseScenarioSample` or standardized JSONL,
4. validate with `python -m coordiworld.cli.validate_data`,
5. run training/evaluation only after official data/API/wrapper is available,
6. generate tables only from real audited JSON/CSV result files.


## Reproducibility Notes

- Do not commit datasets, checkpoints, logs, API keys, or private server paths.
- Do not hardcode experiment results or copy paper table numbers.
- Synthetic smoke metrics are not NAVSIM/OpenScene/Bench2Drive metrics.
- Real NAVSIM/OpenScene results require official data and official metric wrappers.
- `scripts/make_tables_from_results.py` refuses to create a real table without real result JSON/CSV input.

## GitHub Project Safety

`.gitignore` is configured to ignore:

- `/data/`
- `/outputs/`
- `/checkpoints/`
- `/wandb/`
- `.env`
- `.env.*`
- `.idea/`
- `*.pt`, `*.pth`, `*.ckpt`

It uses root-anchored `/data/` so `src/coordiworld/data/` remains trackable.
