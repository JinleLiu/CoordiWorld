# DATA_ADAPTER_SPEC

## 1. BaseScenarioSample structure
`BaseScenarioSample` should be the cross-dataset canonical sample contract containing:

- scenario identity: `scene_id`, `sample_id`, `timestamp`
- coordinate context: `coordinate_frame`, transform metadata
- ego history/future tensors
- agent states/history and optional future labels
- map token payloads / topology payloads
- candidate trajectory set `T_t`
- optional supervision labels for risk/ranking/calibration
- provenance metadata and quality flags

## 2. NAVSIM/OpenScene adapter interfaces
Expected adapter responsibilities:

- Load raw dataset-specific records.
- Convert to `BaseScenarioSample` deterministically.
- Validate required fields and unit conventions.
- Surface missing-field diagnostics (non-silent failure modes).

Suggested interface examples:
- `NavsimAdapter.iter_samples(split: str) -> Iterator[BaseScenarioSample]`
- `OpenSceneAdapter.iter_samples(split: str) -> Iterator[BaseScenarioSample]`

## 3. Synthetic dataset purpose
Synthetic fixtures are used for:

- schema validation,
- adapter dry-run,
- CI-friendly smoke tests,
- deterministic regression tests when real data is unavailable.

Synthetic data must be clearly labeled and must not be presented as benchmark evidence.

## 4. Data path policy
Data roots must come from environment variables, e.g.:

- `DATA_ROOT`
- `NAVSIM_ROOT`
- `OPENSCENE_ROOT`

Adapters must fail with explicit errors when required env vars are missing.

## 5. Path hardcoding prohibition
Hardcoded local absolute paths are forbidden in adapters/config defaults.
