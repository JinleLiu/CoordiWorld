#!/usr/bin/env python3
"""Recompute the C1 paper's frozen saved results on CPU.

The numerical implementation is the unmodified v06 script. This wrapper only
locates, verifies and stages its inputs, then checks the release acceptance
contract. It never loads models, raw logs, native scoring or simulation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / 'paper_revision' / 'replay'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_inputs(data):
    pins = json.loads((REPLAY / 'INPUT_SHA256.json').read_text())
    extra = json.loads((REPLAY / 'RELEASE_INPUT_BINDINGS.json').read_text())
    for name, expected in {**pins, **{k: v for k, v in extra.items() if k != 'recompute_saved_v05.py'}}.items():
        path = data / name
        require(path.is_file(), 'Missing required input: ' + name + '; see paper_revision/DATA_MODELS.md')
        require(digest(path) == expected, 'Input SHA-256 mismatch: ' + name)
    require(digest(REPLAY / 'recompute_saved_v05.py') == extra['recompute_saved_v05.py'],
            'Frozen replay implementation SHA-256 mismatch')
    return pins, extra


def verify_acceptance(data, output):
    import numpy as np
    results = json.loads((output / 'RECALCULATED_COMPARISONS.json').read_text())
    expected = json.loads((data / 'EXPECTED_COMPARISONS.json').read_text())

    # Strict structural comparison supplements the original script's checks.
    def equal(a, b, location):
        if isinstance(b, dict):
            require(isinstance(a, dict) and set(a) == set(b), 'Key mismatch: ' + location)
            for key in b:
                equal(a[key], b[key], location + '.' + key)
        elif isinstance(b, list):
            require(isinstance(a, list) and len(a) == len(b), 'Length mismatch: ' + location)
            for i, (left, right) in enumerate(zip(a, b)):
                equal(left, right, location + '.' + str(i))
        elif isinstance(b, (float, int)):
            require(bool(np.isclose(a, b, rtol=0, atol=1e-10)), 'Numeric mismatch: ' + location)
        else:
            require(a == b, 'Value mismatch: ' + location)

    require(len(results) == len(expected) == 5, 'Expected exactly five frozen comparisons')
    for result, reference in zip(results, expected):
        # Historical expected objects contain additional saved provenance fields.
        equal(result, {key: reference[key] for key in result}, result['comparison'])
    primary = next(x for x in results if x['comparison'] == 'STRATEGY_C1_minus_S1')
    # These constants are acceptance assertions only; all outputs above were
    # calculated by the original script from hash-verified full-precision rows.
    for key, anchor in [('left_Score', 87.21473047728695), ('right_Score', 84.85050801137633),
                        ('scene_mean', 2.3642224659106126),
                        ('drive_cluster_CI95', [1.3342334425852622, 3.6479198659883108])]:
        equal(primary[key], anchor, key)
    require((primary['positive_drives'], primary['negative_drives'], primary['zero_drives']) == (19, 1, 0),
            'Drive sign counts differ')
    for metric, counts in [('NC', (58, 80)), ('TTC', (71, 98)), ('EC', (377, 410))]:
        item = primary['metrics'][metric]
        require((item['left_events'], item['right_events']) == counts, metric + ' event count differs')
    raw = primary['metrics']['rawNC']
    require((raw['left_known'], raw['right_known'], raw['paired_known']) == (1328, 1335, 1328),
            'rawNC unknown denominators changed')

    strategy = read_csv(data / 'STRATEGY_PER_SCENE.csv')
    history = read_csv(data / 'S1_HISTORY_ACCOUNTING.csv')
    require(len(history) == len({r['token'] for r in history}) == 2919, 'Causal input count/uniqueness differs')
    require(set(r['formal'] for r in history) == {'True', 'False'}, 'Unexpected formal flag')
    formal = {r['token']: r['drive'] for r in history if r['formal'] == 'True'}
    require(len(formal) == 1336 and len(history) - len(formal) == 1583, 'Formal/auxiliary accounting differs')
    for method in ['C1', 'S1']:
        group = [r for r in strategy if r['method'] == method]
        require(len(group) == 1336 and {r['token']: r['drive'] for r in group} == formal,
                'Formal token/drive membership differs for ' + method)
    require(len(set(formal.values())) == 20, 'Drive denominator differs')

    # Verify both identities and numerical values of all saved transitions.
    events = read_csv(output / 'RECALCULATED_EVENTS.csv')
    saved = read_csv(data / 'EVENT_TRANSITIONS.csv')
    keys = ('comparison', 'token', 'drive', 'metric', 'transition')
    key = lambda row: tuple(row[k] for k in keys)
    require(len(events) == len(saved) == 509, 'Event transition count differs')
    for left, right in zip(sorted(events, key=key), sorted(saved, key=key)):
        require(key(left) == key(right), 'Event transition identity differs')
        for field in ['left_value', 'right_value']:
            equal(float(left[field]), float(right[field]), 'event.' + field)
    return dict(status='PASS', scope='saved-result CPU recalculation and source-bound acceptance only',
                primary=primary, comparisons=5, formal_scenes_per_policy=1336, drives=20,
                causal_inputs=2919, auxiliary_inputs=1583, event_transitions=509,
                bootstrap_resamples=10000, bootstrap_seed=170018, weighting='equal scene; paired drive clusters',
                tolerance_absolute=1e-10, model_inference=False, native_scoring=False, training=False,
                end_to_end_inference_verified=False, original_replay_unmodified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'paper_revision' / 'numerics' / 'inputs',
                        help='Directory containing the six hash-verified publication input files')
    parser.add_argument('--output', type=Path, required=True, help='New or empty output directory')
    args = parser.parse_args()
    require(not sys.flags.optimize, 'Run without -O: the frozen script uses assertions')
    data, output = args.data_dir.resolve(), args.output.resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'Output directory must be new or empty')
    require(output != data and data not in output.parents, 'Output must be outside the input directory')
    pins, extra = verify_inputs(data)  # Must precede numerical computation and output writes.
    with tempfile.TemporaryDirectory(prefix='coordiworld-saved-replay-') as directory:
        stage = Path(directory)
        (stage / 'scripts').mkdir()
        (stage / 'data').mkdir()
        shutil.copyfile(REPLAY / 'recompute_saved_v05.py', stage / 'scripts' / 'recompute_saved_v05.py')
        for name in [*pins, 'INPUT_SHA256.json']:
            shutil.copyfile(data / name, stage / 'data' / name)
        subprocess.run([sys.executable, '-I', str(stage / 'scripts' / 'recompute_saved_v05.py'),
                        '--output', str(output)], cwd=stage, check=True)
    acceptance = verify_acceptance(data, output)
    acceptance['input_sha256'] = {**pins, **extra}
    (output / 'PAPER_ACCEPTANCE.json').write_text(json.dumps(acceptance, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'comparisons': 5, 'formal_scenes': 1336,
                      'drives': 20, 'causal_inputs': 2919, 'event_transitions': 509}))


if __name__ == '__main__':
    main()
