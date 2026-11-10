# Obtain inputs, models and dependencies

For CPU saved-result recalculation, obtain the numerical attachment identified in [ACCESS.json](ACCESS.json), then run `python scripts/download_paper_inputs.py --archive CoordiWorld_numerics_v12a.zip`. It verifies the whole-archive SHA, the exact member inventory, safe paths and every member SHA before extraction to `paper_revision/numerics/`. The prepared archive contains the six CPU files, seven figure data/metadata files, their source index, a README and a numerical manifest. The original plotting script publication copy is in Git.

**Distribution channel: public GitHub Release.** Download [CoordiWorld_numerics_v12a.zip](https://github.com/JinleLiu/CoordiWorld/releases/download/iotj-revision-v12/CoordiWorld_numerics_v12a.zip) or run `python scripts/download_paper_inputs.py` to retrieve it automatically. The archive is 656739 bytes; SHA-256: `dd95120e73df11653f74672269a92ca7dd7bc5eb4f5dd079b1f51f407c333f0a`. No author credentials or controlled-review access is required. [ACCESS.json](ACCESS.json) holds the downloader fields and the state recorded when its commit was created; the [Release validation summary](https://github.com/JinleLiu/CoordiWorld/releases/tag/iotj-revision-v12) records subsequent anonymous verification.

Full source and three historical configuration copies are in the Git tree; special scene identifiers are externalized into a stable alias configuration. No reverse mapping, original sensor logs, maps, internal communication or connection records are in the numerical attachment. The exact publication file SHAs and frozen provenance are separate identities. Dataset aliases preserve original sorting and grouping; training `hash_rank`, the historical train order and selection basis were not recomputed.

The existing repository contains PNC, IC0 and the frozen world checkpoint parts. Follow `scripts/assemble_world_checkpoint.py` to assemble the existing world parts; `scripts/reproduce.py` checks all 13 model identities. The published revision retains the existing model names and bytes. Original XGBoost model generation used 2.1.4; the lightweight public synthetic evaluation reads exported trees without refitting. The world SHA is `fa04f85c81781a67b60755977c5027d9e0e2490d6d672eac4f4f5ca46e63f3a2`, with `action_conditioned=False` and 300 actor slots.

To run the existing model-integrity check:

```bash
python -m pip install -r requirements-replay.txt
python scripts/reproduce.py --output /path/to/public_checks.json
```

This checks the unchanged weights and executes the six PNC heads on synthetic zero features. It is an integrity/execution test, separate from saved-result statistical recalculation and empirical prediction accuracy.

Original inference additionally requires official [NAVSIM](https://github.com/autonomousvision/navsim), [OpenScene](https://github.com/OpenDriveLab/OpenScene) / nuPlan data and maps, compatible native metric caches, the exact cohorts, predecessor plans and candidate banks. The recorded NAVSIM pin is `0a380a9063d7162ec93d0f51e9990ebac585f720`; local compatibility changes and native EC byte provenance still limit clean rebuilds. Obtain CARLA/Bench2Drive environments and W4/TF++ assets through their respective official projects. Third-party weights, sensor logs, maps and native caches are not bundled.

| Reference | Recorded implementation/weight identity |
|---|---|
| World4Scorer | Commit `7814310e0bd65c6e21f06b81eedfec2ac9a06aa9`; provider download route must be taken from its original acquisition records |
| W4 NAVSIM checkpoint | SHA `22604f83dda8a2bd2dd05356fca8457f0c7dc78439f47db396e5c85d74971a44`; separate from CARLA RouteTP |
| W4 CARLA RouteTP | SHA `ed3f61a9ac850627c07bbf1d37cc0a9e123d5b41a9c1cecefe820644e14aff5b` |
| TF++ single | [carla_garage](https://github.com/autonomousvision/carla_garage) commit `48e6a6dbcb133220333bd65e242fbc2b492bb15b`; checkpoint SHA `d6fbdc28f7398354beadc7cf6765d866457c957f7b470c88ba206e73311a3b44` |

These are preserved identities, not evidence that downloads or native inference were tested in this acceptance. NAVSIM baselines, W4 NAVSIM and W4 CARLA use distinct inputs and checkpoints. Original environment records are references; no full environment image is provided.

Existing `LICENSES/` and `THIRD_PARTY_NOTICES.md` are unchanged. No new repository-wide, data or model license is assigned. External assets remain subject to their providers’ terms.

Recorded fixed-revision NAVSIM and W4-NAVSIM download URLs and file hashes are in [MODEL_ACQUISITION.csv](MODEL_ACQUISITION.csv). These are acquisition records; current reachability and distribution permission were not tested.
