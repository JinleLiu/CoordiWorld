# Recalculate saved results by version

## Unified return numerical supplement

The public [iotj-return-numerics-v01 Release](https://github.com/JinleLiu/CoordiWorld/releases/tag/iotj-return-numerics-v01) provides the author-approved exact archive. Anonymous download and fresh-environment saved-value/figure acceptance passed on 2026-11-17. [ACCESS.json](ACCESS.json) records the current state separately from historical preparation metadata. Manuscript labels such as `final-r12` are editorial versions, not aliases for the earlier repository tag `iotj-revision-v12`.

Exact archive identity: 7243715 bytes, SHA-256 `bd032d49278bb866aa0527bd8ebc061c8a2ac6a3febd9c8d848514c7fd041278`. Distribution filename `CoordiWorld_Return_Numerics_v01.zip` is only an alias of `CoordiWorld_Return_Numerics_v01_candidate.zip`; both designate the same unchanged bytes. Packaging-state messages inside the archive describe its creation state and do not establish subsequent permission, publication or anonymous verification. No internal manifest or scientific file is rewritten by renaming the outer ZIP.

Obtain [CoordiWorld_Return_Numerics_v01.zip](https://github.com/JinleLiu/CoordiWorld/releases/download/iotj-return-numerics-v01/CoordiWorld_Return_Numerics_v01.zip) from the public Release; no author credentials are required. Start in the directory containing the ZIP, outside a repository checkout:

```bash
unzip CoordiWorld_Return_Numerics_v01.zip
cd CoordiWorld_Return_Numerics_v01
python3 -m venv ../reader_env
../reader_env/bin/python -m pip install -r requirements.txt
../reader_env/bin/python -I scripts/reproduce_all.py --output ../reproduced
```

If using the original candidate filename, pass that filename to `unzip`; the extracted folder is identical. The environment and output must stay outside the extracted package, and the output directory must not exist. To check only inventory and SHA identities, add `--verify-only` to the last command. There is no new repository downloader or top-level wrapper for this package.

The unified entry verifies all 78 non-self outer members, including code, documentation and data; the manifest also enumerates 37 nested archive files. It stages supplied inputs, runs the existing analyzers, compares all adopted full-precision values at rtol=0 / atol=1e-10 and generates figures from the recalculated tables. All interval calculations load the sealed draw matrices. Three legacy script copies replace deterministic RNG reconstruction with saved-draw loading; the original files and exact diffs are retained in the package. No scenes, thresholds or intervals are selected again.

Outputs under `../reproduced/` include:

| Subdirectory | Result |
|---|---|
| `pressure/`, `figures/pressure_I.*`, `figures/pressure_E.*` | A0, all five A1 conditions, components/events, current Table I and Fig. 3 |
| `policy_history/` | Original C1/S1 and conditional comparisons, current Table II |
| `systems/` | Full system components, EP, tail and ranking outputs, current Table III |
| `layout/mechanisms/` | All three matched-refit contrasts and diagnostics, current Table IV |
| `history/` | B0/E2, both predecessors and comparators, components, missing denominators and interaction |
| `layout/geometry/` | Frozen geometry strata, GP1/GP2 summary and original numerical plots |
| `closed_loop/` | H/S/V deterministic count checks from minimal real source fields |

The unchanged archive `legacy/CoordiWorld_TableI_Supplementary_Data_and_Code.zip` has a historical TableI filename and now corresponds to **Table III**. H preserves historical study/method summaries and common keys; S/V retain real episode-level projections, with S's prestart failure separate and V's six bridges excluded from 72 confirmations. No H episodes are fabricated from summary counts. The exact minimal inputs are included in the author-approved public archive; this approval does not extend to full raw traces or external assets.

HistorySensitivity's original `SELECTIONS_v01.csv.bank_sha256` remains the feature-NPZ identity. The separate `history/SELECTION_BANK_VIEW_v01.csv` joins it to the true `frozen_pool` SHA. RawNC unknown values are not filled: E2/C1 known counts are 231/96/231 and E2/S1 228/96/229 (CW/GTRS/Simple), with paired support handled separately. See the package dictionary and the [table/figure map](TABLE_FIGURE_MAP.md).

Public-reader acceptance on 2026-11-17 used a fresh macOS arm64 environment: Python 3.9.6, NumPy 1.26.4, pandas 2.2.3, Matplotlib 3.8.4, ReportLab 4.2.5 and pypdfium2 4.30.0, installed from public PyPI with cache disabled. Anonymous archive size/SHA, all outer/nested member identities, saved-value arithmetic, sealed-draw replay, 36 pressure plot cells, 90 geometry plot-data rows and minimal H/S/V counts passed. The original pressure/geometry figures were generated and visually checked. DejaVu Sans and Helvetica were used; no font files are distributed. The Release records the verification scope; no full raw-to-result inference, native scoring or training was run.

## Original public version: iotj-revision-v12

These commands run from the **repository root** and retain their original scope. They do not generate the current common-candidate Fig. 3:

```bash
python3 -m venv .venv-paper
.venv-paper/bin/python -m pip install -r paper_revision/requirements-cpu.txt
.venv-paper/bin/python scripts/download_paper_inputs.py
.venv-paper/bin/python scripts/reproduce_paper.py --output new_saved_output
.venv-paper/bin/python -m pip install -r paper_revision/figures/requirements-figures-v12.txt
.venv-paper/bin/python scripts/reproduce_figures.py --assets-dir paper_revision/numerics/paper_assets --output new_figures
```

Use `scripts/reproduce_paper.py --data-dir INPUT_DIRECTORY --output NEW_DIRECTORY` after installing `paper_revision/requirements-cpu.txt` in a fresh virtual environment. Python 3.9.6 and NumPy 1.26.4 were used for this local release acceptance. The original historical receipt used Python 3.12.14 / NumPy 2.3.5; the new acceptance follows the original requirement `numpy==1.26.4`.

The data directory must contain exactly these required identities (extra files are ignored):

| File | Purpose |
|---|---|
| `STRATEGY_PER_SCENE.csv` | C1 and S1 own-history scores and components, 1,336 each |
| `FOUR_ARM_CONDITIONAL.csv` | Four arms evaluated in the frozen C1 context |
| `EVENT_TRANSITIONS.csv` | All 509 saved transitions across five comparisons |
| `EXPECTED_COMPARISONS.json` | Frozen full-precision comparison and interval assertions |
| `INPUT_SHA256.json` | Actual publication four-input SHA-256 map |
| `S1_HISTORY_ACCOUNTING.csv` | 2,919 causal inputs: 1,336 formal plus 1,583 auxiliary |

Install `CoordiWorld_numerics_v12a.zip` with `python scripts/download_paper_inputs.py --archive ARCHIVE`. The default output is `paper_revision/numerics`; the replay defaults to its `inputs/` subdirectory. The author approved public Release distribution; use `python scripts/download_paper_inputs.py` for the configured anonymous URL, or `--archive` for a manually downloaded copy. The actual channel and SHA are tracked in `ACCESS.json`.

Pins in `replay/INPUT_SHA256.json` and `replay/RELEASE_INPUT_BINDINGS.json` verify the actual publication files. A stable, one-to-one scene/drive mapping preserves the original lexical sort order, method pairing, grouping, predecessor links, formal/auxiliary status and row order. All numerical field strings, timestamps, missing values and expected comparisons are retained; no re-sampling or rounding was used to make publication copies. Original and publication SHAs differ and are separately recorded in `SOURCE_PUBLICATION_BINDINGS_v12a.json`. Private reverse mappings are excluded from Git and the numerical attachment.

The wrapper verifies all inputs and the frozen script, copies the original implementation and four statistical inputs into a temporary isolated layout, and executes `replay/recompute_saved_v05.py` unchanged. It only adds path resolution and acceptance assertions. The original code sorts tokens/drives, resamples paired drive clusters 10,000 times with `default_rng(170018)`, and takes the scene-weighted ratio in each draw. Auxiliary inputs do not enter the main mean.

Outputs are `RECALCULATED_COMPARISONS.json`, `RECALCULATED_DRIVES.csv`, `RECALCULATED_EVENTS.csv`, `CPU_RECALCULATION_RECEIPT.json`, and `PAPER_ACCEPTANCE.json`. Output must be a new or empty directory outside the inputs. Run without `-O`, since the frozen script uses assertions. Any hash or numerical difference fails the command; the fixed absolute tolerance is `1e-10` and relative tolerance is zero.

All five comparisons and event denominators are checked against saved values. In particular, rawNC is known for 1,328 C1 rows, 1,335 S1 rows and 1,328 paired rows. Unknown labels are never filled with zero. The wrapper's numerical anchors are acceptance assertions after real calculation, not output generators.

This entry does not invoke world prediction, candidate generation, controller simulation, native scoring, CARLA or training. It does not establish full inference equivalence. The original implementation and requirements are retained with their own identities. Do not substitute the original native-dependent `analyze_v05.py` for this CPU entry.
