"""Isolated diagnosis of the pinned IDM path endpoint bug; never writes labels.

The optional compatibility probe preserves the original function's successful
returns and only restores a dropped final waypoint after its one-point error.
This is an explicitly modified simulator probe, not an unmodified official run.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from coordiworld.data.openscene import load_log
from coordiworld.navsim_adapter import OfficialComponentScorer


def retry_dropped_endpoint(original, states, events, repair=False):
    try:
        return original(states)
    except AssertionError as error:
        if str(error) != 'Path has to has more than 1 element!':
            raise
        from nuplan.common.actor_state.state_representation import ProgressStateSE2
        from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath
        from nuplan.planning.simulation.path.utils import calculate_progress
        progress = calculate_progress(states)
        retained = [i for i, repeated in enumerate(np.isclose(np.diff(progress), 0.0)) if not repeated]
        event = {'points': [[float(p.x), float(p.y), float(p.heading)] for p in states],
                 'progress': progress, 'legacy_retained_indices': retained, 'repaired': False}
        events.append(event)
        # No invented waypoint, dropped agent, or traffic-policy substitution.
        if (not repair or len(retained) != 1 or len(states) < 2
                or not np.isfinite(progress).all()
                or np.isclose(progress[-1] - progress[retained[0]], 0.0)):
            raise
        indices = retained + [len(states) - 1]
        path = InterpolatedPath([ProgressStateSE2(progress=progress[i], x=states[i].x,
                                  y=states[i].y, heading=states[i].heading) for i in indices])
        event['repaired'] = True
        return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--log-root', required=True)
    parser.add_argument('--map-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scene-ids', nargs='+', required=True)
    parser.add_argument('--repair-probe', action='store_true')
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    import nuplan.planning.simulation.observation.idm.idm_agent as idm_module
    original = idm_module.create_path_from_se2
    events = []
    idm_module.create_path_from_se2 = lambda states: retry_dropped_endpoint(original, states, events, args.repair_probe)
    manifest = Path(args.manifest); data = json.loads(manifest.read_text())
    lookup = {r['scene_id']: r for r in data['scenes']}
    source = Path(__file__).read_bytes(); (output / 'source.py').write_bytes(source)
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'pid': __import__('os').getpid(),
              'repair_probe': args.repair_probe, 'scope': 'Training/validation/calibration simulator diagnosis only; never official label promotion',
              'script_sha256': hashlib.sha256(source).hexdigest(),
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'scene_ids': args.scene_ids, 'scenes': [], 'complete': False,
              'dependency_revisions': {repo: subprocess.check_output(['git', '-C', 'third_party/' + repo, 'rev-parse', 'HEAD'], text=True).strip() for repo in ['navsim', 'nuplan-devkit']}}
    (output / 'plan.json').write_text(json.dumps(report, indent=2) + '\n')
    start = time.monotonic()
    try:
        for token in args.scene_ids:
            row = lookup[token]
            if row['split'] not in ['train', 'validation', 'calibration']:
                raise ValueError('No held-out test diagnosis or method choice')
            log_path = Path(args.log_root) / (row['log_name'] + '.pkl')
            frames = load_log(log_path); indices = {str(f['token']): i for i, f in enumerate(frames)}
            scorer = OfficialComponentScorer(args.map_root, output / 'metric_cache', 'reactive')
            cache = scorer.build_metric_cache(frames, indices[token])
            with np.load(manifest.parent / row['path'], allow_pickle=False) as arrays:
                poses = arrays['input_candidates'][0]; valid = np.flatnonzero(arrays['input_candidate_valid'][0])
            result = {'scene_id': token, 'log_name': row['log_name'], 'split': row['split'],
                      'candidate_sha256': row['candidate_sha256'], 'raw_log_sha256': hashlib.sha256(log_path.read_bytes()).hexdigest(),
                      'scores': {}, 'failures': [], 'path_events': events}
            for candidate in valid:
                try:
                    result['scores'][str(candidate)] = scorer.score(cache, poses[candidate])
                except Exception as error:
                    result['failures'].append({'candidate': int(candidate), 'error': repr(error), 'traceback': traceback.format_exc()})
                finally:
                    gc.enable(); gc.collect()
            result['path_events'] = list(events); events.clear()
            report['scenes'].append(result)
            (output / 'progress.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(json.dumps({'scene_id': token, 'scores': len(result['scores']), 'failures': len(result['failures']), 'path_events': len(result['path_events'])}), flush=True)
        report['complete'] = True
    except Exception:
        report['fatal_traceback'] = traceback.format_exc()
        raise
    finally:
        idm_module.create_path_from_se2 = original
        report['elapsed_seconds'] = time.monotonic() - start
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
