# CoordiWorld: Penalty-Coupled Utility Learning for Structured Trajectory Evaluation

CoordiWorld evaluates structured candidate trajectories using a frozen factual predictor, six learned readout heads, a four-moment projection and candidate eligibility rules. The current paper policy, **C1 (`PNC_H3P_staticGuard_live_v01`)**, adds a current-only stopped-vehicle guard to PNC_H3P; it reuses the frozen world model and readout heads.

The main comparison is C1 versus **S1**, the frozen Simple `A_lexicographic` rule with the same guard, each using its own saved history. The nominal PDM proposer is rule based; shared H3 supply depends on a learned CW shadow choice. See the [method and policy identities](paper_revision/METHOD_CODE.md) for C0/S0 and implementation details.

## Environment and quick start

Use Python 3.9–3.12 with the pinned requirements. From a checkout of this repository:

```bash
python3 -m venv .venv-paper
.venv-paper/bin/python -m pip install -r paper_revision/requirements-cpu.txt
.venv-paper/bin/python scripts/download_paper_inputs.py
```

The downloader retrieves the public [numerical attachment](https://github.com/JinleLiu/CoordiWorld/releases/download/iotj-revision-v12/CoordiWorld_numerics_v12a.zip), checks its SHA and every member, and installs inputs under `paper_revision/numerics/`. No author credentials are required. [Manual installation and model access](paper_revision/DATA_MODELS.md) are also available. The fixed paper version is [iotj-revision-v12](https://github.com/JinleLiu/CoordiWorld/releases/tag/iotj-revision-v12).

## Main result

On the exposed 1,336-scene / 20-drive cohort under the frozen non-reactive protocol:

| C1 score | S1 score | C1 − S1 | Paired drive-cluster 95% CI |
|---:|---:|---:|---:|
| 87.21 | 84.85 | +2.36 | [1.33, 3.65] |

The comparison uses 10,000 paired drive-cluster resamples, seed 170018, and equal scene weighting; 19 drive means are positive and one is negative. This is an evaluation-layer comparison on shared learned supply in this cohort. Full-precision results, event transitions, missing denominators and all five frozen comparisons are retained in the numerical inputs and recalculation output.

## Recalculate statistics and generate figures

```bash
.venv-paper/bin/python scripts/reproduce_paper.py --output new_saved_output
.venv-paper/bin/python -m pip install -r paper_revision/figures/requirements-figures-v12.txt
.venv-paper/bin/python scripts/reproduce_figures.py --assets-dir paper_revision/numerics/paper_assets --output new_figures
```

Use new output directories. The commands verify input and script identities, then run the saved-result calculation and the paper's Python plotting implementation. See [recalculation details](paper_revision/REPRODUCE.md) and the [table/figure input map](paper_revision/TABLE_FIGURE_MAP.md).

**Verified scope:** anonymous acquisition, saved-result CPU recalculation and figure generation passed for the published tag; the [Release validation summary](https://github.com/JinleLiu/CoordiWorld/releases/tag/iotj-revision-v12) records the environment and results. Full raw-data-to-inference, native scoring and training still require external assets and compatible environments and have not been validated end to end. [Execution scope and scientific limits](paper_revision/SCOPE.md) describe those distinctions.

## Documentation

- [Method–code map](paper_revision/METHOD_CODE.md), [source identities](paper_revision/SOURCE_INDEX.csv) and [path/identifier transformations](paper_revision/PUBLICATION_TRANSFORMS.md).
- [Data, models and dependencies](paper_revision/DATA_MODELS.md), including the existing synthetic model-integrity test; [download identity](paper_revision/ACCESS.json).
- [Detailed reproduction](paper_revision/REPRODUCE.md) and [table/figure sources](paper_revision/TABLE_FIGURE_MAP.md).
- [Version changes](paper_revision/CHANGELOG.md) and [historical methods and results](paper_revision/history/INDEX.md), including separate H/S/V evidence.

## Citation and licenses

Use the [manuscript and software citation](CITATION.md). Existing [license files](LICENSES/) and [third-party notices](THIRD_PARTY_NOTICES.md) apply; consult the [data/model access conditions](paper_revision/DATA_MODELS.md) for external assets.
