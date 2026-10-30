# Adopted method and source map

## Main model identity

`config/model_identity.json` gives SHA-256 identities for the unchanged six PNC heads, six IC0 reference heads and reference-world checkpoint. The adopted policy is PNC_H3P from the complete-system freeze; later scalar and joint readout controls do not replace it.

The reference-world checkpoint uses hidden dimension 128, four attention heads, two history layers, feed-forward dimension 512, one rollout block and zero dropout. Its time step is 0.5 s. Its model metadata records a maximum history of eight and maximum actor capacity of 300, provenance and confidence inputs, and **no action conditioning**. The learned residual is combined with kinematic/current-map structure. It is not evidence that a learned reactive multi-agent world was identified from logged trajectories.

The actually consumed decoder is `BodyHeadingKinematicResidual`, which overrides the generic class's legacy state decoder. Its motion-feedback MLP is 10→128→128, and its acceleration/yaw-rate residual MLP is 138→128→2. History acceleration and yaw rate decay with time constant 2 s. The bounded learned corrections are 2 m/s² and 0.2 rad/s, followed by midpoint position integration; the body-heading wrapper retains the observed body/motion-heading offset. All eight motion-feedback/acceleration-head weight and bias tensors are present in the frozen checkpoint. The runtime constructs this subclass and calls `load_state_dict(ck['state_dict'])` with the default strict loading behavior. Checkpoint architecture and runtime dispatch, rather than unused generic decoder defaults, define the adopted model.

Main head input dimensions are Q:698, QP/QTTC/QLK:674, DAC-event:682, raw-NC-event:706. Each head has 150 trees. The four joint heads use a maximum of 15 leaves per tree; event heads use a maximum of seven. The first four values are projected onto `0 <= joint moment <= Q <= 1` before selection. Raw NC uses the event `unfiltered NC < 1`.

The executed tree fits use `eta=.05`, `seed=17`, histogram growth, `lambda=10`, `min_child_weight=5`, and complete row/column sampling. Main QP uses the recorded candidate-relative custom objective (relative-error coefficient one); this coefficient is distinct from XGBoost's L2 regularization value. **A reloaded XGBoost configuration can contain reset training defaults.** `booster_config.json` is used to read inference dimensions/base score; it is not authority for the executed learning rate or seed. Training scripts and explicit configuration are the sources for those parameters.

## Source map

| Component | Source |
|---|---|
| Structured tensor fields, provenance/missing-value semantics | `src/coordiworld/schema.py` |
| OpenScene/current-state serialization and rear-axle pose | `src/coordiworld/data/openscene.py` |
| Current-map serialization | `src/coordiworld/data/maps.py` |
| PDM nominal proposer and 12 perturbations | `src/coordiworld/data/pdm_proposer.py` |
| Generic constant-turn-rate fallback proposer | `src/coordiworld/candidates.py` |
| Generic reference-world architecture | `src/coordiworld/model.py` |
| Actual residual/body-heading wrapper | `src/coordiworld/revision_v5/kinematic_residual.py`, `body_heading.py` |
| Native controller execution | `src/coordiworld/revision_v5/compiled_solve_controller.py` |
| Main feature and candidate pipeline | `research_source/workspace/twentyfourth_20261029/scripts/revision_v5_twentyfourth_prepare.py` |
| Current/history/spatial candidate construction | `research_source/workspace/twentythird_20261029/scripts/revision_v5_twentythird_factory.py`, `revision_v5_twentythird_null_probe.py`, `revision_v5_twentythird_spatial_probe.py` |
| PNC model binding and first-stage selection | `research_source/workspace/twentythird_20261029/scripts/revision_v5_twentythird_policy.py` |
| Appended-candidate eligibility and P0 readout | `research_source/workspace/twentythird_20261029/scripts/revision_v5_twentythird_qualification.py`, `revision_v5_twentythird_progress_readout.py` |
| Main head training | `research_source/workspace/twentysecond_20261028/scripts/revision_v5_twentysecond_train.py`, `revision_v5_twentysecond_physical_risk.py` |
| C/CV/Hybrid representation construction | `research_source/workspace/twentysixth_20261030/scripts/revision_v5_twentysixth_representation.py` |
| A16 matched Direct/F4/Joint experiments | `research_source/readout_study/` |
| Portable main-policy inference | `src/coordiworld_release/policy.py` |

The fallback proposer in `candidates.py` is not the PDM nominal proposer used by the adopted full pipeline. The generic model class contains earlier experimental options; `config/model_identity.json` and the frozen runtime bindings define what was actually adopted. `config/historical_data_adapter.yaml` is preserved for source compatibility; its model/training/calibration defaults are historical settings and do not describe the final experiment.

## Candidate supply and baselines

The base bank is 12 PDM-derived current-state trajectories. A current-only brake repair is used under the recorded contact-risk trigger. On a follow-up frame, if no valid base candidate is EC-compatible with the policy's own prior execution, the pipeline attempts history, current-profile and spatial-profile additions. Duplicates are removed; additions require executed EC greater than zero. History uses a two-second dynamic-profile blend. Passing EC is a comfort result, not a collision proof.

PNC_EC keeps the base supply. PNC_NULL adds the current-profile control. PNC_H3P uses all three supply families with the recorded appended-candidate eligibility rule and known-current-P0 replacement for the progress term. IC0_H3P and Geometry_H3 have their own legal selected histories in the complete-policy comparison. They are not evaluated against a borrowed oracle history.

The representation controls share the current prefix (1,260 columns). CV and Hybrid each append 429 columns: 143 candidate features, the reference candidate's features, and their difference. CV holds observed velocity, heading, dimensions and support fixed and extrapolates position; Hybrid uses the frozen reference world. Source/support and all current-derived columns are shared. The same target, folds, rows, training budget and candidate availability apply to all three.

The A16 readouts compare D150 and D600 scalar trees, F4 factorized heads, Jbox joint moments and Jproj projected moments. A better candidate-level MSE or guaranteed algebraic consistency does not establish higher realized policy utility. The release retains negative and inconclusive comparisons.

## Public-core boundary

The public repository supplies source code, main model weights/configuration and a deterministic synthetic six-head smoke test. It includes no scientific data or aggregate result tables. All saved-evidence numerical replay is supplied separately in the accompanying numerical supplement. The full mechanism-control feature matrices and later portable weights are outside the public-code release and the small supplement; see `RELEASE_SCOPE.md`.
