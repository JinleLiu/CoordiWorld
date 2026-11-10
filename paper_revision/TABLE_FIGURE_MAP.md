# Locate table and figure inputs

Use the saved-result command for the two current A tables. The paper integration team owns the actual manuscript and figure layouts.

| Paper label/asset | Input | Actual implementation | This candidate |
|---|---|---|---|
| `tab:cw_primary` | `STRATEGY_PER_SCENE.csv` and saved event components | `replay/recompute_saved_v05.py` via top-level wrapper | Recalculated and checked |
| `tab:cw_contrasts` | Above plus `FOUR_ARM_CONDITIONAL.csv`, original expected intervals | Same original script, five fixed comparisons | Recalculated and checked |
| Fig. 3 | Approved v12 scripts and full-precision A drive/comparison inputs | Original `plot_saved_results.py`, invoked by `scripts/reproduce_figures.py` | Script in Git; data in author-approved public numerical Release attachment |
| Former Fig. 4 / recorded-control appendix | Original v12 script and B saved control window | Original `plot_saved_results.py`, invoked by `scripts/reproduce_figures.py` | Same plot algorithm; route identifier alias only |
| `tab:b-closed-loop-v07`, difference table | H `all_episodes.csv` + study-specific common_keys; S `FINAL_v01.json`; V `FINAL_v01.json` | B:F0058/F0059, B:F0498 + confirmation_statistics, B:F2222 + v05_statistics | Bytes/source acquired, no new B recalculation |
| `tab:b-h-strong-references-v07` | H saved study rows, missing official records preserved | Original H analysis and v07 exports | Historical reference; source indexed |
| `tab:cw_public_context` | Sealed A historical baseline table | Existing baseline source bindings and A saved exports | Historical, not substituted for C1/S1 |

For B, S has 216 confirmation rows and V has 72 confirmation rows plus six bridge rows. Do not combine H/S/V or replace strict success with Completed. The recorded-control window follows each policy's own history near first plan divergence; it is not a shared counterfactual or collision-time geometry.

Install `figures/requirements-figures-v12.txt`, then run `python scripts/reproduce_figures.py --assets-dir paper_revision/numerics/paper_assets --output new_figures` from the repository. Eight external data/index pins in `figures/INPUT_SHA256.json` and the Git script pin in `figures/CODE_SHA256.json` are checked before execution. The wrapper stages the original layout and invokes the sole paper-team implementation. Only the route identifier text has a publication alias; the wrapper contains no plotting logic.

| Figure input in numerical attachment | Role |
|---|---|
| `A_drive_effects_v07.csv` | Saved drive effects, original row/order/values |
| `A_drive_mapping_v07.csv` | D01–D20 display labels to stable public drive aliases |
| `A_paired_effects_v07.csv` | Saved paired effects and intervals |
| `B_CASE_CONTROL_TIMELINE_v07.csv` | Each policy's own recorded control sequence and time anchors |
| `B_CASE_OUTCOMES_v07.csv` | Case outcomes; missing/failed observations retained |
| `B_CASE_SEALED_EPISODE_ROWS_v07.json` | Sealed case metadata, paths/route alias only |
| `B_CASE_WINDOW_PROVENANCE_v07.json` | Window definition and source provenance, paths/route alias only |
| `SOURCE_FILES.json` | Actual publication SHA values and separate original SHA values |

The final-layout acceptance used Python 3.9.6, NumPy 1.26.4, Matplotlib 3.8.4 and installed DejaVu Sans. The paper source package recorded Python 3.12.4 with the same NumPy/Matplotlib versions. Saved data, coordinates, ordering, time anchors and plotting parameters were checked; only route text is aliased. Five regenerated PDFs are rendered for visual QA. Font files are not distributed. PDF/SVG creation metadata and cross-environment PNG pixels need not be byte-identical.

The numerical attachment is approved for public Release distribution. See `ACCESS.json` for the exact URL and hash, and the Release notes for anonymous post-publication verification. Original internal paper assets and private reverse maps are excluded.
