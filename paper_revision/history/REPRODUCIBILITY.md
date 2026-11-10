# Entry points

- Public code/weights: install `requirements-replay.txt` and run `python scripts/reproduce.py` for 13 model-hash checks and the synthetic six-head smoke test. No study numerical data is included or needed.
- Accompanying numerical supplement: overlay its files on the public repository and run `python scripts/replay_supplement.py --output supplement_reproduction.json` for the previously validated larger core saved-evidence replay. Its entry point is separate from the public script.
- Native research: inspect `src/` and `research_source/`. Execution from official raw logs/maps requires the compatible native environment and upstream artifacts. Full mechanism-control refitting is not turnkey in the public-code package.

`config/model_identity.json` records unchanged weights and the active BodyHeadingKinematicResidual decoder. Legacy generic residual settings are retained as checkpoint metadata and do not override the executed decoder. The frozen world checkpoint is an inference dependency; a complete earlier pretraining lineage is not claimed.
