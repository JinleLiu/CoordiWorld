# CoordiWorld Project Audit

## Audit Date
2026-04-21

## Current Implemented Modules

- `src/coordiworld/scene_summary/`
  - `SceneSummary` dataclass schema.
  - JSON/dict I/O.
  - deterministic validator.
  - ICA-style helper modules for transforms, association, fusion, and generator.
- `src/coordiworld/data/`
  - canonical `BaseScenarioSample` and `ScenarioLabels`.
  - deterministic synthetic dataset.
  - shared candidate pool with nominal, speed-scaled, lateral-shift, and curvature-perturbed variants.
  - dependency-light collate helpers.
  - standardized JSONL adapter.
  - native dataset adapter boundaries for NAVSIM, OpenScene, nuScenes, and Waymo.
- `src/coordiworld/tokens/`
  - SceneSummary tokenizer.
  - action/candidate trajectory tokenizer.
  - map token feature encoder.
- `src/coordiworld/models/`
  - minimal PyTorch structured rollout model.
  - temporal scene encoder.
  - action-conditioned rollout and prediction heads.
- `src/coordiworld/risks/`
  - geometry helpers.
  - collision risk.
  - map-grounded rule-violation risk.
  - predictive uncertainty.
  - calibration interface.
  - final lower-is-better score `J`.
- `src/coordiworld/evaluation/`
  - ranking metrics.
  - calibration metrics.
  - NAVSIM metric adapter dry-run stub.
  - robustness perturbations and ranking stability.
- `src/coordiworld/attribution/`
  - entity token masking.
  - nearby entity selection.
  - counterfactual post-hoc attribution helpers.
- `src/coordiworld/training/`
  - Stage I losses.
  - one-batch synthetic Stage I smoke trainer.
  - fixed pairwise ranking batch schema for Stage II.
- `scripts/`
  - environment checker.
  - Stage I/Stage II/evaluation synthetic dry-run scripts.
  - result table generator that refuses fake benchmark tables.

## Current Missing or Incomplete Modules

- Full real-data NAVSIM/OpenScene/native parser implementation.
- Official NAVSIM metric wrapper integration.
- Full Stage II pairwise ranking training loop.
- Full experiment runner for multi-seed real-data evaluation.
- Real-data calibration fitting workflow.
- Real-data table generation from audited benchmark outputs.
- Official reproduction of paper benchmark metrics.

## Current CLI List

- `python -m coordiworld.cli.validate_data`
  - validates `synthetic` and standardized `jsonl` adapters.
  - returns clear errors for native datasets without root/dependencies.
- `python -m coordiworld.cli.build_scene_summary`
  - supports synthetic/jsonl smoke validation.
  - native real-data conversion remains a dry-run/TODO boundary.
- `python -m coordiworld.cli.train_stage1`
  - points to synthetic smoke script; full real-data training is not implemented.
- `python -m coordiworld.cli.train_stage2`
  - points to pairwise schema dry-run; full ranking trainer is not implemented.
- `python -m coordiworld.cli.calibrate`
  - interface placeholder; no fake calibration result.
- `python -m coordiworld.cli.evaluate`
  - points to synthetic evaluation dry-run; no fake NAVSIM metrics.
- `python -m coordiworld.cli.run_ablation`
  - placeholder; no fake benchmark metrics.

## Current Tests

- SceneSummary schema and validation tests.
- ICA-style fusion/generator synthetic tests.
- candidate pool and synthetic dataset tests.
- tokenizer shape/mask/slot tests.
- rollout model CPU smoke tests.
- risk scoring tests.
- Stage I loss and smoke training tests.
- pairwise ranking batch schema tests.
- ranking/calibration metric tests.
- attribution/robustness tests.
- data registry, JSONL adapter, validate-data CLI, and README smoke-command tests.
- import, CLI help, and environment-config tests.

## Current Data Interfaces

- `synthetic`
  - deterministic in-memory dataset for tests and smoke commands.
- `jsonl`
  - standardized intermediate format for converted real-data samples.
- `navsim`
  - adapter boundary using `NAVSIM_ROOT` or `config.root`; no fake samples.
- `openscene`
  - adapter boundary using `OPENSCE_ROOT` or `config.root`; no fake samples.
- `nuscenes`
  - optional SceneSummary/ICA source boundary using `NUSCENES_ROOT`; no fake samples.
- `waymo`
  - optional SceneSummary/ICA source boundary using `WAYMO_ROOT`; no fake samples.

## Current Runnable Commands

```bash
python -m coordiworld.cli.validate_data --dataset synthetic --max-samples 2
python -m coordiworld.cli.validate_data --dataset jsonl --config configs/datasets/jsonl_example.yaml --max-samples 2
python -m coordiworld.cli.build_scene_summary --dataset synthetic --max-samples 1
bash scripts/run_stage1_synthetic.sh
bash scripts/run_stage2_synthetic.sh
bash scripts/run_eval_synthetic.sh
python scripts/make_tables_from_results.py --help
python -m pytest -q
ruff check src tests
```

## Content That Must Not Be Claimed Complete

- Real NAVSIM EPDMS or official NAVSIM benchmark results.
- Real OpenScene training or evaluation results.
- Real Bench2Drive results.
- Real nuScenes/Waymo benchmark results.
- Official paper table reproduction.
- Any result table generated without real audited JSON/CSV inputs.
- Any claim that synthetic smoke metrics are benchmark metrics.

## README Cleanup Priorities

- State clearly that CoordiWorld is a fixed candidate-set evaluator, not a trajectory generator.
- Describe implemented modules and status without overclaiming.
- Provide install, environment, synthetic smoke, JSONL, and native adapter commands.
- Document that `SceneSummary` is structured world state, not natural-language summary.
- Explain that real benchmark results require official data/API/wrapper.
- Keep License as `TBD` unless a license file is added by the project owner.

## Data Adapter Cleanup Priorities

- Keep `synthetic` and `jsonl` fully runnable without real datasets.
- Keep native adapters strict: validate root, validate official dependency, then raise TODO.
- Do not hardcode server paths.
- Use environment-variable placeholders in configs.
- Preserve clear errors: `DataRootError`, `MissingDependencyError`, `DatasetFormatError`.
