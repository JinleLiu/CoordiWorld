# Locate table and figure inputs

Use the saved-result command for the two current A tables. The paper integration team owns the actual manuscript and figure layouts.

| Paper label/asset | Input | Actual implementation | This candidate |
|---|---|---|---|
| `tab:cw_primary` | `STRATEGY_PER_SCENE.csv` and saved event components | `replay/recompute_saved_v05.py` via top-level wrapper | Recalculated and checked |
| `tab:cw_contrasts` | Above plus `FOUR_ARM_CONDITIONAL.csv`, original expected intervals | Same original script, five fixed comparisons | Recalculated and checked |
| Fig. 3 | Approved v12 scripts and full-precision A drive/comparison inputs | Original `plot_saved_results.py`, invoked by `scripts/reproduce_figures.py` | Received, SHA-verified and regenerated; assets held outside Git for distribution review |
| Former Fig. 4 / recorded-control appendix | Original v12 script and B saved control window | Original `plot_saved_results.py`, invoked by `scripts/reproduce_figures.py` | Received and regenerated using the same original script |
| `tab:b-closed-loop-v07`, difference table | H `all_episodes.csv` + study-specific common_keys; S `FINAL_v01.json`; V `FINAL_v01.json` | B:F0058/F0059, B:F0498 + confirmation_statistics, B:F2222 + v05_statistics | Bytes/source acquired, no new B recalculation |
| `tab:b-h-strong-references-v07` | H saved study rows, missing official records preserved | Original H analysis and v07 exports | Historical reference; source indexed |
| `tab:cw_public_context` | Sealed A historical baseline table | Existing baseline source bindings and A saved exports | Historical, not substituted for C1/S1 |

For B, S has 216 confirmation rows and V has 72 confirmation rows plus six bridge rows. Do not combine H/S/V or replace strict success with Completed. The recorded-control window follows each policy's own history near first plan divergence; it is not a shared counterfactual or collision-time geometry.

Install `figures/requirements-figures-v12.txt`, then run `python scripts/reproduce_figures.py --assets-dir /path/to/approved_paper_assets --output /path/to/new_figures` from the repository. The nine script/data/index pins in `figures/INPUT_SHA256.json` are checked before execution. The wrapper stages the exact original file layout and calls the paper team's original script; it contains no plotting logic.

This acceptance used Python 3.9.6, NumPy 1.26.4, Matplotlib 3.8.4 and the installed Matplotlib DejaVu Sans font. The source package recorded Python 3.12.4 with the same NumPy/Matplotlib versions. All saved data checks and plotting parameters match; all five SVG geometry/text sequences match exactly. PDF pages are rendered for visual QA. Font files are not distributed. PDF/SVG creation metadata need not be byte-identical.

The original script, seven data/metadata inputs and source map are preserved unchanged in `local_review_only/paper_assets/`. That directory contains dataset identifiers and internal provenance paths and is excluded from the public Git patch pending review. Its inclusion in the local ZIP does not approve public distribution.
