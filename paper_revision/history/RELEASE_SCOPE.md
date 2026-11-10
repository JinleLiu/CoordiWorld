# Public code release and separate numerical supplement

The public repository contains source code, policy/model configurations, frozen world and PNC/IC0 readout weights, and a deterministic synthetic smoke test. It does not distribute individual scene/drive records, per-mapping rows or plans, empirical feature fixtures, or aggregate result tables. `scripts/reproduce.py` checks weight hashes and executes six readout heads on synthetic zero features. It does not recompute benchmark predictions, confidence intervals or risk losses.

All scientific numerical replay evidence is supplied separately in the accompanying author numerical supplement. Its `scripts/replay_supplement.py` uses the anonymous numerical inputs needed for the larger core replay already validated locally: a real twelve-candidate main fixture, three-cohort saved mapping results, drive event sufficient statistics/transitions and 1,280 selected-risk anchor rows. Overlay the supplement on the public code repository to share its main weights and portable inference code. This saved-evidence replay remains distinct from rerunning native simulation or reconstructing all mappings from raw per-scene execution.

The public synthetic test is not a substitute accuracy experiment and does not regenerate or expose any excluded real-data record. The complete TRAIN240 feature matrices and later portable control weights are outside the public code release and the small numerical supplement. Research scripts require the appropriate data and artifacts; they are not turnkey C/CV/Hybrid or Direct/F4/Joint retraining commands.

The frozen predictor is an inference dependency. Its checkpoint identity and active decoder source do not establish recovery of its complete earlier training lineage.
