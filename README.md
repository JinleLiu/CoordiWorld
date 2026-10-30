# CoordiWorld

Source code and main model weights for the revised CoordiWorld structured trajectory evaluator. The adopted policy is `PNC_H3P`, combining a shared structured forecast, six readout heads, controller-executed candidates and history-compatible candidate supply.

## Check the public code and weights on CPU

```bash
python -m pip install -r requirements-replay.txt
python scripts/reproduce.py
```

The entry point verifies 12 unchanged PNC/IC0 model hashes and the frozen world-checkpoint hash, then runs all six PNC heads on deterministic **synthetic zero features**. It writes `public_checks.json`. This is an execution smoke test, not a benchmark or prediction-accuracy result. It requires no study dataset, empirical feature fixture or numerical result table.

The world checkpoint is assembled automatically from the distributed parts when its complete file is absent, and its complete SHA-256 is checked. The smoke test does not execute the world forecast through PyTorch or invoke native simulation.

## Public contents

- `src/`: structured inputs, forecast, candidate/controller implementation and portable tree evaluation.
- `research_source/`: original research and mechanism-control training scripts for inspection and adaptation.
- `models/PNC/`, `models/IC0/`, `models/world/`: main/comparison readout weights and the frozen forecast checkpoint.
- `config/`: model identity and recorded policy/training specifications.

**All scientific numerical replay materials are supplied separately in the accompanying numerical supplement.** The public repository contains no individual study records, per-mapping results/plans, real-data feature fixtures, or aggregate result tables. To use the supplement, overlay its files on this repository and run `python scripts/replay_supplement.py --output supplement_reproduction.json`. Its independent entry point checks the real main-head fixture and saved three-cohort mapping/bootstrap, event and selected-risk evidence.

The public code and small numerical supplement do not include all later TRAIN240 feature matrices or portable mechanism-control weights. C/CV/Hybrid control refitting is not a turnkey command. Running the native research pipeline from raw logs requires official NAVSIM/OpenScene data, maps, compatible dependencies and upstream artifacts. Releasing the frozen predictor's checkpoint does not establish its complete earlier pretraining lineage.

See [release scope](docs/RELEASE_SCOPE.md), [data](docs/DATA.md), [evaluation](docs/EVALUATION.md), and [source map](docs/METHOD_AND_CODE.md). Third-party notices are preserved in `LICENSES/` and `THIRD_PARTY_NOTICES.md`.
