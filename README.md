# CoordiWorld: Penalty-Coupled Utility Learning for Structured Trajectory Evaluation

CoordiWorld evaluates structured trajectories using a frozen factual predictor, six learned readout heads, a four-moment projection and candidate eligibility rules. The deployed **C1 (`PNC_H3P_staticGuard_live_v01`)** adds a current-only stopped-vehicle guard to PNC_H3P and reuses its frozen models. S1 is the frozen Simple `A_lexicographic` selector with the same guard. See the [method–code map](paper_revision/METHOD_CODE.md) for C0/S0, learned upstream dependencies and the separate matched-refit arms.

## Key results

| Protocol and predecessor | Scenes / drives | Score difference | Paired-drive 95% CI |
|---|---:|---:|---:|
| Original shared proposal pipeline; each policy propagates its own history: C1 − S1 | 1,336 / 20 | +2.36 | [1.33, 3.65] |
| External common E2 candidates; common recorded C1 predecessor: CW − GTRS | 232 / 20 | +9.16 | [5.41, 13.00] |
| The same 232 scenes and fixed E2 candidates; common recorded S1 predecessor: CW − GTRS | 232 / 20 | +8.51 | [5.00, 12.19] |

These are related protocols on an exposed cohort, not three independent test sets or complete-system SOTA comparisons. The predecessor-source interaction is −0.65 [−2.01, 0.80]; it does not establish equivalence. Current **Fig. 3** shows common-candidate expansion; the earlier drive-effect figure is **Supplementary Fig. S1**. [Table/figure mapping](paper_revision/TABLE_FIGURE_MAP.md) locates all four main tables, mechanism, geometry and sensitivity results.

## Install and reproduce saved results

The unified return supplement contains the current statistics and Fig. 3 entry point. Its exact file list awaits distribution approval; there is currently no public or controlled download for that supplement. [Access identity](paper_revision/ACCESS.json) and [reproduction instructions](paper_revision/REPRODUCE.md) distinguish it from the existing public version.

Given an approved copy of the archive, extract it and run **inside its `CoordiWorld_Return_Numerics_v01/` directory** (Python 3.9–3.12):

```bash
python3 -m venv ../reader_env
../reader_env/bin/python -m pip install -r requirements.txt
../reader_env/bin/python -I scripts/reproduce_all.py --output ../reproduced
```

Keep the environment and new output directory outside the package. `scripts/reproduce_all.py` belongs to the numerical package, not this repository root. It verifies the manifest, recalculates saved values using frozen draws and runs the original pressure/geometry plotting scripts. The supplement passed local reader acceptance; public-URL acceptance has not occurred.

The fixed [iotj-revision-v12 Release](https://github.com/JinleLiu/CoordiWorld/releases/tag/iotj-revision-v12) remains public and unchanged. Its repository-root commands reproduce the original policy-history/conditional results and historical plots; see [legacy commands](paper_revision/REPRODUCE.md#original-public-version-iotj-revision-v12). Saved-result checks do not reproduce model inference, native scoring, simulation or training; those require the [external assets and environments](paper_revision/DATA_MODELS.md).

## Documentation, citation and terms

[Method and source identities](paper_revision/METHOD_CODE.md) · [table/figure inputs](paper_revision/TABLE_FIGURE_MAP.md) · [data and models](paper_revision/DATA_MODELS.md) · [validation scope](paper_revision/SCOPE.md) · [version changes](paper_revision/CHANGELOG.md) · [historical results](paper_revision/history/INDEX.md).

Use the [manuscript/software citation](CITATION.md). Existing [license files](LICENSES/) and [third-party notices](THIRD_PARTY_NOTICES.md) remain applicable; no new license is assigned.
