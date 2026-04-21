# SCENESUMMARY_SPEC

## 1. Definition
SceneSummary is a **structured world state representation**, not a natural-language summary.
It is the canonical, deterministic machine interface for CoordiWorld inputs.

## 2. Top-level schema (recommended)
- `scene_id`
- `timestamp`
- `coordinate_frame`
- `ego`
- `agents`
- `map_tokens`
- `provenance`
- `metadata`

## 3. Agent field schema (recommended)
Each agent record should include at least:

- `id`
- `type`
- `x`, `y`, `yaw`, `vx`, `vy`
- `length`, `width`
- `confidence`
- `covariance_xy`
- `existence_prob`
- `source_modalities`
- `source_ids`
- `fusion_lineage`
- `ambiguity_flags`
- `semantic_attributes`

All numeric fields must be bound to `coordinate_frame` conventions and unit definitions.

## 4. ICA-style deterministic generator requirements
The SceneSummary generator should implement ICA-style multi-source fusion with explicit stages:

1. coordinate normalization,
2. LiDAR/BEVFusion geometry seed matching,
3. radar velocity attachment,
4. camera projection-based semantic attachment,
5. continuous attribute fusion,
6. categorical voting,
7. ambiguity flag generation.

The pipeline MUST be deterministic and reproducible under fixed input snapshots.

## 5. Explicit prohibitions
The following are forbidden:

- Directly promoting camera-only 2D detections to CoordiWorld agents when reliable 3D geometry is absent.
- Calling any LLM service inside SceneSummary generator logic.
- Overwriting structured geometric facts using natural-language synopsis.
