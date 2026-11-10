# CoordiWorld: Penalty-Coupled Utility Learning for Structured Trajectory Evaluation

## How to use

For the current C1 paper's saved-result CPU recalculation, create a Python 3.9–3.12 environment and install the original pinned requirement:

```bash
python3 -m venv .venv-paper
.venv-paper/bin/python -m pip install -r paper_revision/requirements-cpu.txt
.venv-paper/bin/python scripts/reproduce_paper.py --data-dir /path/to/approved_saved_inputs --output /path/to/new_output
```

The six full-precision input files currently accompany the **local review candidate**, outside the Git patch. Their derived data and identifiers still require an author distribution decision. No public download is claimed. If approved inputs are later installed in `paper_revision/data/`, `--data-dir` can be omitted. The entry point verifies their SHA-256 identities before calculation. See [recalculation instructions](paper_revision/REPRODUCE.md) and [data/model access](paper_revision/DATA_MODELS.md).

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

Fig. 3 and the former Fig. 4 use the paper integration team's exact v12 plotting implementation. It has been regenerated in a fresh directory; all five SVGs retain identical plotted geometry and text. The original assets remain in the local review companion pending distribution review.

```bash
python -m pip install -r paper_revision/figures/requirements-figures-v12.txt
python scripts/reproduce_figures.py --assets-dir /path/to/approved_paper_assets --output /path/to/new_figures
```

The wrapper verifies inputs and runs the original plotting script unchanged. See [paper table/figure mapping](paper_revision/TABLE_FIGURE_MAP.md).

[Version changes](paper_revision/CHANGELOG.md), [release-note candidate](paper_revision/RELEASE_NOTES_CANDIDATE.md), [citation candidate](CITATION.md), and [known scope](paper_revision/SCOPE.md) describe this revision. The earlier README and results remain accessible through [historical documentation](paper_revision/history/INDEX.md). Existing third-party notices and license files are preserved; this candidate assigns no new license.
