# Public execution checks and separate numerical replay

`python scripts/reproduce.py` checks 13 unchanged model-file SHA-256 identities and runs all six PNC heads on deterministic synthetic zero features. It checks that the outputs and policy scores are finite and that a valid candidate is selected. No scientific data or reported numerical result is needed. The synthetic test establishes executable model-loading/forward behavior only; it is not a benchmark, accuracy estimate or evidence of planning quality.

All scientific numerical replay uses the separately supplied accompanying supplement and its `scripts/replay_supplement.py`. That previously validated local core calculation checks a real main-head fixture, saved mapping summaries, paired whole-drive resampling, event sufficient statistics, and selected-anchor risk losses/bins. The public repository alone does not recalculate these quantities or distribute their input/result tables.

Neither the smoke test nor the numerical supplement is a new simulator run, a new model fit, an independent road-safety test or reconstruction of the frozen predictor's entire pretraining lineage.
