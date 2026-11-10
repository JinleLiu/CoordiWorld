# Recalculate saved paper results

Use `scripts/reproduce_paper.py --data-dir INPUT_DIRECTORY --output NEW_DIRECTORY` after installing `paper_revision/requirements-cpu.txt` in a fresh virtual environment. Python 3.9.6 and NumPy 1.26.4 were used for this local release acceptance. The original historical receipt used Python 3.12.14 / NumPy 2.3.5; the new acceptance follows the original requirement `numpy==1.26.4`.

The data directory must contain exactly these required identities (extra files are ignored):

| File | Purpose |
|---|---|
| `STRATEGY_PER_SCENE.csv` | C1 and S1 own-history scores and components, 1,336 each |
| `FOUR_ARM_CONDITIONAL.csv` | Four arms evaluated in the frozen C1 context |
| `EVENT_TRANSITIONS.csv` | All 509 saved transitions across five comparisons |
| `EXPECTED_COMPARISONS.json` | Frozen full-precision comparison and interval assertions |
| `INPUT_SHA256.json` | Original four-input SHA-256 map |
| `S1_HISTORY_ACCOUNTING.csv` | 2,919 causal inputs: 1,336 formal plus 1,583 auxiliary |

Pins are supplied in `replay/INPUT_SHA256.json` and `replay/RELEASE_INPUT_BINDINGS.json`. Files retain original bytes, token/drive pairing, timestamps, row order and missing values. They have not been anonymized, re-sampled, rounded or reconstructed from the paper. Distribution remains pending; the local acceptance companion is not an approved public dataset.

The wrapper verifies all inputs and the frozen script, copies the original implementation and four statistical inputs into a temporary isolated layout, and executes `replay/recompute_saved_v05.py` unchanged. It only adds path resolution and acceptance assertions. The original code sorts tokens/drives, resamples paired drive clusters 10,000 times with `default_rng(170018)`, and takes the scene-weighted ratio in each draw. Auxiliary inputs do not enter the main mean.

Outputs are `RECALCULATED_COMPARISONS.json`, `RECALCULATED_DRIVES.csv`, `RECALCULATED_EVENTS.csv`, `CPU_RECALCULATION_RECEIPT.json`, and `PAPER_ACCEPTANCE.json`. Output must be a new or empty directory outside the inputs. Run without `-O`, since the frozen script uses assertions. Any hash or numerical difference fails the command; the fixed absolute tolerance is `1e-10` and relative tolerance is zero.

All five comparisons and event denominators are checked against saved values. In particular, rawNC is known for 1,328 C1 rows, 1,335 S1 rows and 1,328 paired rows. Unknown labels are never filled with zero. The wrapper's numerical anchors are acceptance assertions after real calculation, not output generators.

This entry does not invoke world prediction, candidate generation, controller simulation, native scoring, CARLA or training. It does not establish full inference equivalence. The original implementation and requirements are retained with their own identities. Do not substitute the original native-dependent `analyze_v05.py` for this CPU entry.
