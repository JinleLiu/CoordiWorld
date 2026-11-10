# Obtain inputs, models and dependencies

For CPU saved-result recalculation, use the six files listed in [REPRODUCE.md](REPRODUCE.md). They are original full-precision numerical exports, with scene/drive identifiers and paired history structure. They are held outside Git pending an author decision on distribution terms and identifier handling. No public or reviewer download URL has been assigned. The code fails with the precise missing filename if inputs are absent.

The existing repository contains PNC, IC0 and the frozen world checkpoint parts. Follow `scripts/assemble_world_checkpoint.py` to assemble the existing world parts; `scripts/reproduce.py` checks all 13 model identities. This candidate neither renames nor replaces model bytes. Original XGBoost model generation used 2.1.4; the lightweight public synthetic evaluation reads exported trees without refitting. The world SHA is `fa04f85c81781a67b60755977c5027d9e0e2490d6d672eac4f4f5ca46e63f3a2`, with `action_conditioned=False` and 300 actor slots.

Original inference additionally requires official [NAVSIM](https://github.com/autonomousvision/navsim), [OpenScene](https://github.com/OpenDriveLab/OpenScene) / nuPlan data and maps, compatible native metric caches, the exact cohorts, predecessor plans and candidate banks. The recorded NAVSIM pin is `0a380a9063d7162ec93d0f51e9990ebac585f720`; local compatibility changes and native EC byte provenance still limit clean rebuilds. Obtain CARLA/Bench2Drive environments and W4/TF++ assets through their respective official projects. Third-party weights, sensor logs, maps and native caches are not bundled.

| Reference | Recorded implementation/weight identity |
|---|---|
| World4Scorer | Commit `7814310e0bd65c6e21f06b81eedfec2ac9a06aa9`; provider download route must be taken from its original acquisition records |
| W4 NAVSIM checkpoint | SHA `22604f83dda8a2bd2dd05356fca8457f0c7dc78439f47db396e5c85d74971a44`; separate from CARLA RouteTP |
| W4 CARLA RouteTP | SHA `ed3f61a9ac850627c07bbf1d37cc0a9e123d5b41a9c1cecefe820644e14aff5b` |
| TF++ single | [carla_garage](https://github.com/autonomousvision/carla_garage) commit `48e6a6dbcb133220333bd65e242fbc2b492bb15b`; checkpoint SHA `d6fbdc28f7398354beadc7cf6765d866457c957f7b470c88ba206e73311a3b44` |

These are preserved identities, not evidence that downloads or native inference were tested in this acceptance. NAVSIM baselines, W4 NAVSIM and W4 CARLA use distinct inputs and checkpoints. Original environment records are references; no full environment image is provided.

Existing `LICENSES/` and `THIRD_PARTY_NOTICES.md` are unchanged. No new repository-wide, data or model license is selected. Being present in an internal handoff does not authorize redistribution.

Recorded fixed-revision NAVSIM and W4-NAVSIM download URLs and file hashes are in [MODEL_ACQUISITION.csv](MODEL_ACQUISITION.csv). These are acquisition records; current reachability and distribution permission were not tested.
