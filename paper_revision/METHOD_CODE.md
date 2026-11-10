# Inspect the method and its source

Start with `selection/` for the three small NumPy components. `SOURCE_INDEX.csv` maps every imported author source alias to its original SHA, delivered SHA, publication-copy SHA and exact local destination. Full-file source and extracted components remain distinct.

| Module/result | Actual source | Scope |
|---|---|---|
| C1 final selector | [frozen_cw_selector.py](selection/frozen_cw_selector.py) | Unmodified AST extraction of original lines 89–99; only component import checked |
| Simple S0/S1 | [simple_rules.py](selection/simple_rules.py) | Frozen A_lexicographic; S1 applies the same guard on Simple-valid candidates |
| Current-only guard | [current_static_geometry_v01.py](selection/current_static_geometry_v01.py) | Stopped vehicle geometry; no future or learned inputs |
| Full C1 original entry | `A:F0112` in [SOURCE_INDEX.csv](SOURCE_INDEX.csv); full file in local review companion | Original --policy static_guard; full source held locally because it contains a scene identifier |
| C0 input fallback | `A:F0109` in [SOURCE_INDEX.csv](SOURCE_INDEX.csv); full file in local review companion | Original environment; includes the recorded auxiliary fallback |
| S1 own-history pipeline | `A:F0077` in [SOURCE_INDEX.csv](SOURCE_INDEX.csv); full file in local review companion | Full source held locally; uses its own committed history |
| Six-head inference | [pnc_inference_v02.py](source/A_R18-research-20261031T1047/code/pnc_inference_v02.py) | FrozenPNC; unchanged PNC models |
| Projection and features | [revision_v5_twentysecond_primary.py](source/workspace/twentysecond_20261028/scripts/revision_v5_twentysecond_primary.py) | Four joint moments and x674 feature construction |
| Eligibility | [revision_v5_twentythird_policy.py](source/workspace/twentythird_20261029/scripts/revision_v5_twentythird_policy.py) | DAC and physical-NC eligibility; original anchor semantics |
| H3 timing and P0 | [revision_v5_twentythird_timing_active.py](source/A_R18-research-20261031T1047/code/peer_reference/revision_v5_twentythird_timing_active.py) | Learned shadow choice in current-profile supply; original appended-slot semantics |
| Nominal proposer | [pdm_proposer.py](source/workspace/twentyfourth_20261029/runtime_src/coordiworld/data/pdm_proposer.py) | PDMClosedPlanner source; differs from generic CTRV fallback |
| Actor serialization | [openscene.py](source/workspace/twentyfourth_20261029/runtime_src/coordiworld/data/openscene.py) | Observed fields, derived masks and missing-confidence handling |
| Structured schema | [schema.py](source/workspace/twentyfourth_20261029/runtime_src/coordiworld/schema.py) | Source/lineage fields are adapter semantics, not verified V2X provenance |
| World architecture | [model.py](source/workspace/twentyfourth_20261029/runtime_src/coordiworld/model.py) | Checkpoint identity, not unused historical YAML defaults, governs the architecture |
| Actual residual decoder | [body_heading.py](source/workspace/twentyfourth_20261029/runtime_src/coordiworld/revision_v5/body_heading.py) | BodyHeadingKinematicResidual; factual action-blind model |
| Captured graph execution | [captured_factual_v2.py](source/workspace/twentyfourth_20261029/runtime_src/coordiworld/revision_v5/captured_factual_v2.py) | graph_encode / graph_states; source inspection only |
| Controller native kernel | [compiled_controller.cpp](source/A_R15-research-20261030T0013/source/compiled_controller.cpp) | Recovered original C++ identity; no compilation in this acceptance |
| Historical adapter settings | [group_cache_v2.yaml](source/A_R15-research-20261030T0013/peer_B_R14/config/group_cache_v2.yaml) | Only historical data/candidate adapter evidence; not current model training parameters |
| H saved-statistics entry | [revision_v5_twentyeighth_results.py](source/B_H/code/revision_v5_twentyeighth_results.py) | Main34 and ParkingExit4 stay separate; no new H calculation |
| S final saved analysis | [analyze_confirmation.py](source/workspace/r18_followup_20261102T030813Z/code/analyze_confirmation.py) | Final full-archive analysis, not the earlier S source snapshot |
| S frozen active agent | [cw_followup_agent.py](source/workspace/r18_followup_20261102T030813Z/versions/development_v02/cw_followup_agent.py) | versions/development_v02 is the actual confirmation agent |
| S Simple/CW readout | [cw_common_policy.py](source/workspace/r18_followup_20261102T030813Z/versions/development_v02/cw_common_policy.py) | Supply-v2 common interface; must not replace H/V platform |
| S route-cluster statistics | [confirmation_statistics.py](source/workspace/r18_followup_20261102T030813Z/code/confirmation_statistics.py) | 216 confirmation rows, 3 strata; original 97.5% primary intervals |
| V final analysis export | [analyze_v05_export_fix_v01.py](source/workspace/r18_followup_20261102T030813Z/core_contribution_v05/code/analyze_v05_export_fix_v01.py) | Original export fix preserves statistics and avoids duplicate slot keyword |
| V route-cluster statistics | [v05_statistics.py](source/workspace/r18_followup_20261102T030813Z/core_contribution_v05/code/v05_statistics.py) | 72 confirmation rows, 6 bridge rows separate; 50,000 draws historically |

`config/adopted_policy.json` records current identities and original freeze hashes. Existing `../config/model_identity.json` remains the model identity authority for the unchanged PNC/IC0/world bytes; its historical `primary_method=PNC_H3P` identifies C0, not the added C1 guard.

Source copies preserve full function bodies and historical branches. Path-only substitutions are recorded separately; `<CW_WORKSPACE>`, `<CW_ARCHIVE_ROOT>` and existing `${CW_ARCHIVE_ROOT}` markers are inspection references, not a newly verified runtime path resolver. Core original entries containing scene identifiers are available in the local review companion with explicit SHA bindings. They are not silently anonymized or replaced.

The selector extraction is not a complete predictor or pipeline. Its inputs already include real head outputs, base eligibility, execution proxies and a bank receipt. The pre-guard winner is only the pre-guard decision in that supplied context, not proof of C0 own-history equivalence. Import/AST checks do not validate full inference equivalence.

Simple tie selection has a documented historical discrepancy: the frozen protocol said 1e-12 while the executed developer used B>A+1e-10. Saved A/B choices and scores were identical on all 160 development scenes, so both tolerances select A. Neither the executed code nor frozen results are changed.

Third-party native scorer implementations are indexed in `EXTERNAL_SOURCES.csv` and acquired upstream. Existing license notices continue to apply to adaptations. Files with the same logical source identity and identical byte hash are shared; different versions are retained in `source/versions/` or separate H/S/V locations. `SOURCE_CONFLICTS.csv` lists conflicts without choosing the newest version.
