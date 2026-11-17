# Locate current tables and figures

The mappings below were checked against the effective final-r12 LaTeX sources and their actual asset paths. Figure/table numbers are the checked return-manuscript numbering; manuscript identifiers such as `final-r12` do not rename the repository's `iotj-revision-v12` tag. See [access state and working directories](REPRODUCE.md) before running commands.

## Unified return supplement

All paths in this table are relative to the extracted **`CoordiWorld_Return_Numerics_v01/` numerical package**, not the repository. Run its `scripts/reproduce_all.py`; the individual analyzers below are staged by that entry. The exact package file list is approved for public distribution; see ACCESS.json and the independent Release for availability and anonymous verification.

| Manuscript identity | Package inputs | Actual package implementation and output |
|---|---|---|
| Table I(a/b), `tab:cw_pressure_e2`; Supplement S3, Tables S4–S7 | `pressure/inputs/PER_SCENE_v01.csv`, `BOOTSTRAP_DRAWS_v01.npz`, sealed summaries and events | `scripts/pressure_saved.py`; `reproduced/pressure/` contains all A0/five-condition comparisons and components |
| **Fig. 3**, `fig:cw_pressure` | Freshly recalculated `B_RECALCULATED_SUMMARY_v01.csv` and `B_PAIRED_COMPARISONS_v01.csv`, staged with the original plotting filenames | `scripts/plot_pressure.py`; `reproduced/figures/pressure_I.pdf` and `pressure_E.pdf`; no smoothing or new sampling |
| Table II, `tab:cw_matched` / `tab:cw_primary` / `tab:cw_contrasts`; Supplement S2 | `policy_history/data/STRATEGY_PER_SCENE.csv`, `FOUR_ARM_CONDITIONAL.csv`, expected values and event identities | `policy_history/scripts/recompute_saved_v05.py`; original own-history and fixed-context contrasts remain distinct |
| **Table III**, `tab:cw_public_context`; Supplement S1 | `legacy/CoordiWorld_TableI_Supplementary_Data_and_Code.zip`, preserved byte-for-byte | `scripts/systems_saved.py`; `reproduced/systems/` retains all 11 complete configurations and the separate PNC-bank diagnostic; TableI in the filename is historical |
| Table IV, `tab:cw_mechanisms`; Supplement S4, Tables S16–S18 | `mechanisms/inputs/`, `CORE_PROTOCOL_v01.json`, sealed draw matrix and expected contrasts | `scripts/mechanisms_saved.py`; `reproduced/layout/mechanisms/CORE_RESULTS_v01.csv`; four matched-refit arms, all three contrasts, not new deployed C1 weights |
| Section IV-B; Supplement S3.1, `sec:s_history`, Tables S8–S15 | `history/inputs/`, `SOURCE_MANIFEST_v01.json`, `SELECTION_BANK_VIEW_v01.csv` | `scripts/history_saved.py`; `reproduced/history/`; B0/E2, both common predecessors, both comparators and full components/rawNC support |
| Section IV-F; Supplement S5, Tables S19–S22 | `geometry/attributes/`, frozen protocol, legacy system tables and draws; `geometry/expected/GEOMETRY_PRESSURE_SUMMARY_v01.csv` | `scripts/geometry_saved.py`, `scripts/plot_geometry.py`; all strata, primary summary and numerical plots in `reproduced/layout/geometry/` |
| Supplement S6, Tables S23–S28 | `closed_loop/inputs/H_studies.json`, `S_FINAL_v01.json`, `V_FINAL_v01.json`; frozen count export | `scripts/closed_counts.py`; `reproduced/closed_loop/COUNTS_BY_SEED.csv`; H keeps summary granularity, S/V retain real episode fields |

GP1 is keyed by `(margin:narrow, C1, PDM-Closed, DAC)` and GP2 by `(curvature_nonintersection:high, C1, World4Scorer-v2-lam1, LK)` in `geometry/expected/PAIRED_STATS_v01.csv`. Their primary summary retains the original **97.5%** intervals; those are distinct from the paired table's 95% intervals. GP2 remains an operational sampled-route proxy, with exact continuous-path/current-ego intersection status unverified.

The original pressure data, full components, adverse changes, rawNC unknowns and negative/cross-zero mechanism outcomes remain. HistorySensitivity preserves `bank_sha256` as the candidate-feature NPZ SHA; the separate view supplies the actual frozen-pool SHA. H-main and the four ParkingExit routes remain separate; S has 216 confirmations and a separate prestart failure; V has 72 confirmations and six excluded bridges. Strict success and Completed are not interchangeable.

## Historical public plots and current supplementary use

The unchanged [iotj-revision-v12 Release](https://github.com/JinleLiu/CoordiWorld/releases/tag/iotj-revision-v12) supplies the following data via `CoordiWorld_numerics_v12a.zip`. The repository's `scripts/reproduce_figures.py` invokes `paper_revision/figures/scripts/plot_saved_results.py` with its SHA-checked data.

| Current content identity | Historical artifact and input | Scope |
|---|---|---|
| **Supplementary Fig. S1**, `fig:cw_drive_effects` (main-document reference `supp-fig:cw_drive_effects`) | `CW_Fig3_drive_effects_v12.pdf`; `A_drive_effects_v07.csv`, `A_drive_mapping_v07.csv`, `A_paired_effects_v07.csv` | Original 20-drive plot and overall C1−S1 interval. Its historical Fig3 filename is retained; it is not current main Fig. 3. |
| Supplementary Fig. S2, `fig:cw_event_counts` / `fig:cw_paired_effects` | Manuscript composition `CW_supporting_statistics_v10.pdf`; penalty and paired-contrast values remain in the policy-history tables | Panel values are recalculated by the saved-result entry. The manuscript's composed v10 layout is not claimed as an output of the v12 plotting wrapper or the new package. |
| Recorded-control appendix, formerly main Fig. 4 | `B_CASE_CONTROL_TIMELINE_v07.csv`, `B_CASE_OUTCOMES_v07.csv`, sealed metadata/provenance and `SOURCE_FILES.json`; original `CW_controls_*_v12` plots | Each policy's own recorded history near first plan divergence; not a shared counterfactual, collision-time geometry or a new CARLA execution. |

The eight legacy data/index pins and plotting-code pin remain unchanged. The former drive/comparison figure identity remains accessible in the fixed tag's [historical map](https://github.com/JinleLiu/CoordiWorld/blob/iotj-revision-v12/paper_revision/TABLE_FIGURE_MAP.md). The new supplement does not change or relabel any frozen numerical experiment.
