# Attribution and license scope

This release contains CoordiWorld research source, derived scientific artifacts, and portable numerical-reproduction utilities. No new repository-wide license has been assigned to author-owned code or model parameters. Public availability does not itself assign an open-source license to material that lacked one in the supplied artifacts; the repository owner can designate that license separately.

## NAVSIM

The native evaluation protocol, PDM nominal proposer, state conventions, and controller-related adaptations use [autonomousvision/navsim](https://github.com/autonomousvision/navsim), pinned in the experiments to commit `0a380a9063d7162ec93d0f51e9990ebac585f720`. Its original Apache-2.0 license and copyright notice are preserved in `LICENSES/navsim/LICENSE`.

CoordiWorld modules calling or adapting that stack include `data/pdm_proposer.py`, `revision_v5/compiled_controller.py`, `revision_v5/compiled_solve_controller.py`, and `revision_v5/native_adapter.py`. These adaptations add batch/controller interfaces, coordinate handling and scientific instrumentation. They do not change the upstream license. External native dependencies are installed separately rather than vendored here.

## nuPlan devkit and data

The [nuPlan devkit](https://github.com/motional/nuplan-devkit) supplies state, map and vehicle interfaces used by the native stack. Its supplied notice is preserved in `LICENSES/nuplan-devkit/LICENSE.txt` (Motional, 2021; Apache-2.0).

OpenScene/nuPlan data are obtained under their own licenses; the code license does not override the data terms. Raw sensor logs and maps are not redistributed here. Derived numerical features, identifiers and evaluation outputs originate from those datasets and remain subject to applicable upstream terms. See [Data access](docs/DATA.md).

## Other dependencies

NumPy, pandas, XGBoost, PyTorch, SciPy, Shapely, PyYAML and pyquaternion are external dependencies and retain their respective licenses. No vendored XGBoost package or shared library is included. Model parameter exports are not the XGBoost implementation.

## Release changes

Research source retains its original module names. Account-specific absolute workspace paths were replaced with `COORDIWORLD_WORKSPACE` lookups. Public-facing introductory docstrings were clarified. The new portable replay independently expresses the recorded score arithmetic and tree inference. Main PNC, IC0 and reference-world parameter bytes are unchanged and checked by SHA-256.
