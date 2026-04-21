# Data Interface

CoordiWorld uses one canonical evaluator sample type: `BaseScenarioSample`. All dataset adapters should produce this structure or fail with a clear error.

## Core Contracts

Defined in `src/coordiworld/data/base.py`:

- `BaseScenarioSample`
- `ScenarioLabels`
- `FutureAgentState`
- `ScenarioDataset` / `DatasetAdapter`
- `DatasetSplit`
- `DataRootError`
- `MissingDependencyError`
- `DatasetFormatError`

Required `BaseScenarioSample` fields:

- `scene_id`
- `timestamp`
- `scene_summary_history`
- `candidate_trajectories`
- `logged_ego_future`
- `labels`
- `metadata`

Labels include:

- `collision`
- `violation`
- `pseudo_sim_score`
- `progress`

## Adapter Registry

Use `src/coordiworld/data/registry.py`:

```python
from coordiworld.data.registry import available_datasets, build_dataset

print(available_datasets())
dataset = build_dataset("synthetic", {"num_samples": 2})
```

Registered names:

- `synthetic`
- `jsonl`
- `navsim`
- `openscene`
- `nuscenes`
- `waymo`

## Synthetic Adapter

Config: `configs/datasets/synthetic.yaml`

The synthetic adapter never reads real data. It creates small deterministic `SceneSummary` histories, shared candidate trajectories, future-agent fixtures, and labels.

```bash
python -m coordiworld.cli.validate_data --dataset synthetic --max-samples 2
```

## Standardized JSONL Adapter

Config: `configs/datasets/jsonl_example.yaml`

Each JSONL line is one scenario sample. The adapter validates:

- required top-level fields,
- `SceneSummary` schema,
- `[M,H,3]` candidate trajectories,
- `[H,3]` logged future,
- labels,
- optional `future_agents`.

Minimal example:

```bash
python -m coordiworld.cli.validate_data --dataset jsonl --config configs/datasets/jsonl_example.yaml
```

Required JSONL fields:

- `scene_id`
- `timestamp`
- `scene_summary_history`
- `candidate_trajectories`
- `logged_ego_future`
- `labels`

Optional JSONL fields:

- `sample_id`
- `coordinate_frame`
- `future_agents`
- `transform_metadata`
- `provenance`
- `quality_flags`
- `metadata`

## Native Real-Data Adapters

Native adapters are strict boundaries. They do not fabricate samples.

### NAVSIM

- File: `src/coordiworld/data/navsim_adapter.py`
- Config: `configs/datasets/navsim.yaml`
- Root: `NAVSIM_ROOT` or `config.root`
- Missing root: `DataRootError`
- Missing official package: `MissingDependencyError`

### OpenScene

- File: `src/coordiworld/data/openscene_adapter.py`
- Config: `configs/datasets/openscene.yaml`
- Root: `OPENSCE_ROOT` or `config.root`
- Missing root: `DataRootError`
- Missing official package: `MissingDependencyError`

### nuScenes

- File: `src/coordiworld/data/nuscenes_adapter.py`
- Config: `configs/datasets/nuscenes.yaml`
- Root: `NUSCENES_ROOT` or `config.root`
- Missing devkit: `MissingDependencyError`

### Waymo

- File: `src/coordiworld/data/waymo_adapter.py`
- Config: `configs/datasets/waymo.yaml`
- Root: `WAYMO_ROOT` or `config.root`
- Missing official package: `MissingDependencyError`

## Environment Variables

```bash
DATA_ROOT=/path/to/datasets
NAVSIM_ROOT=/path/to/datasets/navsim
OPENSCE_ROOT=/path/to/datasets/opensce
NUSCENES_ROOT=/path/to/datasets/nuscenes
WAYMO_ROOT=/path/to/datasets/waymo
OUTPUT_ROOT=/path/to/coordiworld/outputs
CHECKPOINT_ROOT=/path/to/coordiworld/checkpoints
WANDB_MODE=offline
```

Do not commit real values. Use `.env.example` only as a template.

## Shared Candidate Pool

`src/coordiworld/data/candidate_pool.py` implements the protocol-level candidate families:

- nominal proposal,
- speed-scaled variants,
- lateral-shift variants,
- curvature-perturbed variants.

All compared evaluators should receive the same candidate set per scene/time step.
