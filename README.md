# CoordiWorld: Penalty-Coupled Utility Learning for Structured Trajectory Evaluation

## How to use

For the current C1 paper's saved-result CPU recalculation, use Python 3.9–3.12 and the pinned requirements. The v12a numerical attachment is technically prepared and tested, but its distribution is **awaiting an explicit author decision**. There is no public or controlled download yet. [ACCESS.json](paper_revision/ACCESS.json) records the actual status and attachment SHA; a local path or hash is not an access entitlement.

Once you have the authorized `CoordiWorld_numerics_v12a.zip`, install it with the hash-checking entry point:

```bash
python3 -m venv .venv-paper
.venv-paper/bin/python -m pip install -r paper_revision/requirements-cpu.txt
.venv-paper/bin/python scripts/download_paper_inputs.py --archive CoordiWorld_numerics_v12a.zip
.venv-paper/bin/python scripts/reproduce_paper.py --output new_saved_output
```

The default installed input directory is `paper_revision/numerics/inputs`. Without `--archive`, the acquisition command requires an approved and verified URL in `ACCESS.json` and currently stops with an explicit unavailable-access message. The proposed destination is a numerical Release attachment; neither that proposal nor deidentification grants distribution rights. See [recalculation instructions](paper_revision/REPRODUCE.md) and [data/model access](paper_revision/DATA_MODELS.md).

For source inspection, start with the [method–code map](paper_revision/METHOD_CODE.md) and [source identities](paper_revision/SOURCE_INDEX.csv). The existing public model-integrity test remains available:

```bash
python -m pip install -r requirements-replay.txt
python scripts/reproduce.py --output /path/to/public_checks.json
```

That test checks unchanged weights and executes the six PNC heads on synthetic zero features. It is an integrity/execution test, with no empirical prediction-accuracy result.

## Current method and identities

CoordiWorld evaluates structured candidate trajectories using a shared frozen factual predictor, six independently fitted readout heads, a four-moment projection, candidate eligibility and history-compatible supply. The adopted paper policy adds the existing current-only stopped-vehicle guard to PNC_H3P.

| Label | Exact identity | History in the main comparison |
|---|---|---|
| C1 | `PNC_H3P_staticGuard_live_v01` | C1's own saved history |
| C0 | Original `PNC_H3P` | C1 context in the conditional four-arm table; historical policy results remain separate |
| S0 | Frozen Simple `A_lexicographic` | C1 context in the conditional table |
| S1 | The same Simple with the same guard operator | S1's own saved history |

C1 reuses the frozen world and PNC heads. It is not a newly trained world model. Existing `models/PNC`, `models/IC0` and `models/world` retain their names and bytes. The nominal PDM proposer is rule based; the shared H3 current-profile supply still depends on a learned CW shadow choice. A head-independent Simple final ranking does not make the upstream system learning free.

## Paper result and replay

On the exposed 1,336-scene / 20-drive source cohort under the frozen non-reactive protocol:

| C1 Score | S1 Score | C1 − S1 | Paired drive-cluster 95% interval |
|---:|---:|---:|---:|
| 87.21473047728695 | 84.85050801137633 | 2.3642224659106126 | [1.3342334425852622, 3.6479198659883108] |

The command above recomputes this and all four conditional contrasts from saved rows with 10,000 paired drive-cluster resamples, seed 170018, and equal scene weighting. Nineteen drive means are positive and one is negative. NC event counts are 58/80, TTC 71/98 and EC 377/410 (C1/S1). Raw-NC unknown values remain unknown. Event transitions, missing denominators and adverse changes are retained in the output.

The result measures an evaluation-layer increment on shared learned supply in this tested cohort. Independent generalization, real-world safety, SOTA and isolated module benefits are not established. H/S/V closed-loop evidence remains separate, including negative results and recorded failures.

## Execution levels and figures

| Entry | Available scope |
|---|---|
| `scripts/reproduce_paper.py` | Tested saved-result CPU recalculation; no model or native scorer calls |
| `paper_revision/selection/` and source map | Inspectable frozen method components; CPU import and source-identity checks |
| Original C0/C1/S1 and H/S/V entries | Located historical implementations; require their own original assets and environment |
| Clean-environment raw-to-result inference or training | Not validated |

Fig. 3 and the former Fig. 4 use the paper integration team's sole approved v12 Python plotting implementation. Its complete publication copy is now in Git; only route identifier strings were replaced by a stable alias. The seven full-precision data/metadata files and source index are in the proposed numerical attachment.

```bash
.venv-paper/bin/python -m pip install -r paper_revision/figures/requirements-figures-v12.txt
.venv-paper/bin/python scripts/reproduce_figures.py --assets-dir paper_revision/numerics/paper_assets --output new_figures
```

The wrapper verifies the actual publication data and script SHA values. A fresh-layout local run preserved plotted coordinates, ordering, time anchors and text meanings; no new plotting implementation, smoothing or intervals were introduced. See [table/figure mapping](paper_revision/TABLE_FIGURE_MAP.md).

The seven previously isolated complete source files and three historical freeze copies are included in this candidate. Identifier-dependent special branches read [source_identifiers_v12a.json](paper_revision/config/source_identifiers_v12a.json); explicit logical roots resolve through a small standard-library helper. Complete source, extracted portable components and tested saved-statistic entries have distinct validation scopes. See [publication transformations](paper_revision/PUBLICATION_TRANSFORMS.md).

[Version changes](paper_revision/CHANGELOG.md), [release-note candidate](paper_revision/RELEASE_NOTES_CANDIDATE.md), [citation candidate](CITATION.md), and [known scope](paper_revision/SCOPE.md) describe this revision. The earlier README and results remain accessible through [historical documentation](paper_revision/history/INDEX.md). Existing third-party notices and license files are preserved; this candidate assigns no new license.
