"""Generate current frozen-M0 TRAIN banks and true labels on the predeclared 40 scenes.

The original PNC infer function is used intact except for returning its already
computed diagnostic arrays. Candidate generation receives four past/current
frames only. The separately called evaluator is the inherited TRAIN reactive
protocol. This script performs no fitting or outcome-based scene selection.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import json
import os
import pickle
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path('${CW_ARCHIVE_ROOT}')
RUN = ROOT / 'A_R18-followup-v03-20261102T1447'
R18 = ROOT / 'A_R18-research-20261031T1047'
sys.path[:0] = [str(R18 / 'code'), str(R18 / 'deps')]
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
import torch
import xgboost as xgb
import pnc_inference_v02 as pnc
from coordiworld.navsim_adapter import OfficialComponentScorer
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score

METRICS = ['no_at_fault_collisions', 'drivable_area_compliance',
           'driving_direction_compliance', 'traffic_light_compliance',
           'ego_progress', 'time_to_collision_within_bound', 'lane_keeping', 'history_comfort']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=default, allow_nan=False) + '\n')
    tmp.replace(path)


class FrozenOOF(pnc.ns['PNC']):
    def __init__(self, fold):
        self.models = {}
        self.bindings = {}
        for head in ['q', 'qp', 'qttc', 'qlk', 'dac_event', 'physical_nc']:
            path = RUN / f'models/M0_oof/fold{fold}/{head}.ubj'
            booster = xgb.Booster()
            booster.load_model(path)
            booster.set_param({'nthread': 1})
            self.models[head] = booster
            self.bindings[head] = dict(path=str(path), sha256=sha(path), dimensions=booster.num_features())

    def predict(self, a, ec=None):
        self.last_prediction = super().predict(a, ec)
        return self.last_prediction


def diagnostic_infer():
    source = R18 / 'code/peer_reference/revision_v5_twentythird_timing_active.py'
    tree = ast.parse(source.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'infer')
    node = copy.deepcopy(original)
    assert isinstance(node.body[-1], ast.Return)
    original_return = node.body[-1].value
    assert isinstance(original_return, ast.Tuple) and len(original_return.elts) == 3
    extra = ast.parse("(asset, base['current'], ec, native, j if supply else None)", mode='eval').body
    node.body[-1].value = ast.Tuple(elts=original_return.elts + extra.elts, ctx=ast.Load())
    assert ast.dump(ast.Module(body=node.body[:-1], type_ignores=[])) == ast.dump(ast.Module(body=original.body[:-1], type_ignores=[]))
    namespace = dict(pnc.ns)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), 'exec'), namespace)
    return namespace['infer'], dict(source=str(source), source_sha256=sha(source),
             unchanged_numerical_body=True, change='return already-computed assets, current state, EC proxy, native executions and brake slot only')


def native_owner():
    import nuplan.planning.simulation.observation.idm.idm_agent as idm
    from diagnose_idm_path import retry_dropped_endpoint
    retries = []
    original_path = idm.create_path_from_se2
    idm.create_path_from_se2 = lambda poses: retry_dropped_endpoint(original_path, poses, retries, True)
    owner = OfficialComponentScorer(pnc.m.ROOT / 'data/maps', RUN / 'data/metric_cache', 'reactive')
    source = next((pnc.m.PEER / 'source').rglob('revision_v5_twentythird_native.py'))
    node = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.ClassDef) and x.name == 'Capture')
    ns = {'METRICS': METRICS}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
    return owner, ns['Capture'](owner.scorer), retries


def score_candidate(owner, capture, cache, poses):
    capture.calls = []
    table, states = pdm_score(cache, Trajectory(poses, owner.prediction_sampling),
                             owner.sampling, owner.simulator, owner.scorer, owner.traffic)
    assert capture.calls and capture.calls[0]['role'] == 'candidate'
    filtered = np.array([float(table.iloc[0][k]) for k in METRICS])
    raw = np.array([float(capture.calls[0]['raw'][k]) for k in METRICS])
    assert np.isfinite(filtered).all() and np.isfinite(raw).all()
    return filtered, raw, states, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', choices=['benchmark', 'full'], default='benchmark')
    args = ap.parse_args()
    assert args.targets == 'benchmark', 'Full alignment requires the recorded timing/coverage decision first'
    started = time.monotonic()
    cfg = json.loads((RUN / 'configs/DATA_ALIGNMENT_FREEZE_v01.json').read_text())
    label_cfg = json.loads((RUN / 'configs/LABEL_PROTOCOL_FREEZE_v01.json').read_text())
    assert label_cfg['training_traffic'].startswith('reactive')
    natural = json.loads((RUN / 'evidence/TRAIN_SOURCE_INDEX.json').read_text())['rows']
    wanted = set(cfg['benchmark_tokens'])
    by_log = {}
    for row in natural:
        by_log.setdefault(row['raw_log'], []).append(row)
    model_manifest = json.loads((RUN / 'configs/M0_OOF_MODEL_MANIFEST.json').read_text())
    for entry in model_manifest['models']:
        assert sha(RUN / entry['remote_relative']) == entry['sha256']
    write(RUN / 'QUEUE_STATUS.json', dict(at=now(), status='RUNNING', stage='benchmark_model_loading', pid=os.getpid(), targets=40, fits=0))
    models = {f: FrozenOOF(f) for f in range(3)}
    pipe = pnc.m.Pipeline()
    infer, infer_binding = diagnostic_infer()
    owner, capture, idm_retries = native_owner()
    load_seconds = time.monotonic()-started
    write(RUN / 'evidence/BENCHMARK_BINDINGS.json', dict(at=now(), script_sha256=sha(__file__),
          inherited_IDM_compatibility_sha256=sha(Path(__file__).parent/'diagnose_idm_path.py'),
          data_freeze_sha256=sha(RUN/'configs/DATA_ALIGNMENT_FREEZE_v01.json'),
          label_freeze_sha256=sha(RUN/'configs/LABEL_PROTOCOL_FREEZE_v01.json'),
          world_sha256=pipe.checkpoint_sha256, original_runtime_bindings=pnc.m.BINDINGS+pnc.bindings,
          infer=infer_binding, models={str(f):m.bindings for f,m in models.items()},
          load_seconds=load_seconds, original_old_main_deadlines_invoked=False, new_fits=0))
    completed, failures, timings = [], [], []
    native_calls = 0
    inference_calls = 0
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('per-record180s guard')))
    for log, target_rows in by_log.items():
        if not any(r['token'] in wanted for r in target_rows):
            continue
        raw_bytes = Path(log).read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        assert all(r['raw_log_sha256'] == digest for r in target_rows)
        raw = pickle.loads(raw_bytes)
        all_targets = {r['current_index']:r for r in target_rows}
        # Use the COMPLETE frozen natural roster's connected history segments,
        # even in this timing batch, so later full alignment can reuse exact histories.
        indices = sorted({j for i in all_targets for j in range(max(3, i-3), i+1)})
        segments = []
        for i in indices:
            if not segments or i != segments[-1][-1]+1 or not 0<int(raw[i]['timestamp'])-int(raw[i-1]['timestamp'])<550000:
                segments.append([])
            segments[-1].append(i)
        fold = target_rows[0]['fold']
        model = models[fold]
        for segment in segments:
            requested = [i for i in segment if i in all_targets and all_targets[i]['token'] in wanted]
            if not requested:
                continue
            segment = [i for i in segment if i <= max(requested)]
            previous = previous_actual = previous_ts = None
            segment_started = time.monotonic()
            for i in segment:
                token = str(raw[i]['token'])
                is_target = token in wanted
                dest = RUN / 'data/current_M0' / token
                dest.mkdir(parents=True, exist_ok=True)
                before = time.monotonic()
                try:
                    signal.alarm(180)
                    write(RUN/'QUEUE_STATUS.json', dict(at=now(),status='RUNNING',stage='benchmark_generate_and_label',pid=os.getpid(),completed_targets=len(completed),target_total=40,current=token,inference_calls=inference_calls,native_calls=native_calls,failures=len(failures)))
                    h = raw[i-3:i+1]
                    assert len(h)==4 and str(h[-1]['token'])==token
                    assert np.allclose(np.diff([int(f['timestamp'])for f in h])/1e6, .5, atol=.08)
                    history_id = dict(raw_log_sha256=digest, segment_start_token=str(raw[segment[0]]['token']),
                                      previous_token=None if previous_ts is None else str(raw[i-1]['token']),
                                      previous_timestamp_us=previous_ts, previous_plan_sha256=None if previous is None else array_sha(previous), fold=fold)
                    if (dest/'receipt.json').exists():
                        raise RuntimeError('Existing output requires explicit audited resume; no overwrite')
                    t = time.monotonic()
                    selected, a, rec, asset, current, proxy_ec, executions, brake_slot = infer(pipe,model,h,digest,previous,previous_ts)
                    generation_seconds = time.monotonic()-t
                    inference_calls += 1
                    count = len(a['valid'])
                    origins = ['native_PDM']*min(12,count)
                    if brake_slot is not None:
                        origins[int(brake_slot)]='brake_supply'
                    origins += ['H3_'+x['variant'] for x in rec['variant_steps'] if x['admitted']]
                    assert len(origins)==count
                    a['candidate_origin']=np.array(origins)
                    a['candidate_id']=np.array([token+':'+array_sha(p) for p in a['candidates']])
                    a['EC']=np.full(count,np.nan) if proxy_ec is None else np.asarray(proxy_ec)
                    np.save(dest/'own_predicted_execution.npy',selected)
                    np.save(dest/'trajectory.npy',a['candidates'][rec['choice']])
                    # Preserve banks at all natural target frames reached by this prefix.
                    if i in all_targets:
                        np.savez_compressed(dest/'candidate_features.npz',**a)
                        np.savez_compressed(dest/'world_assets.npz',**asset)
                        np.savez_compressed(dest/'head_outputs.npz',**model.last_prediction)
                        np.save(dest/'all_predicted_executions.npy',executions)
                        write(dest/'current.json',current)
                    t = time.monotonic()
                    cache = owner.build_metric_cache(raw,i)
                    cache.dump()
                    cache_seconds = time.monotonic()-t
                    valid=np.asarray(a['valid'],bool)
                    to_score=np.flatnonzero(valid) if is_target else np.array([rec['choice']])
                    components=np.full((count,8),np.nan);physical=np.full((count,8),np.nan)
                    actual_ec=np.full(count,np.nan);mask=np.zeros(count,bool)
                    states_by_candidate={}
                    t=time.monotonic()
                    for cid in to_score:
                        comp,phys,states,table=score_candidate(owner,capture,cache,a['candidates'][cid])
                        components[cid]=comp;physical[cid]=phys;mask[cid]=True;states_by_candidate[int(cid)]=states
                        table.to_pickle(dest/f'native_candidate_{cid}.pkl')
                        np.save(dest/f'execution_{cid}.npy',states)
                        native_calls+=1
                        if previous_actual is not None:
                            offset=round((int(h[-1]['timestamp'])-previous_ts)/1e5)
                            actual_ec[cid]=pnc.ns['ec_matrix'](states[None],previous_actual,offset)[0]
                    label_seconds=time.monotonic()-t
                    np.savez_compressed(dest/'native_labels.npz',components=components,raw_components=physical,
                                        rawNC=physical[:,0],EC=actual_ec,labeled=mask,valid=valid,candidate_id=a['candidate_id'])
                    previous=selected;previous_actual=states_by_candidate[int(rec['choice'])];previous_ts=int(h[-1]['timestamp'])
                    np.save(dest/'own_actual_execution.npy',previous_actual)
                    receipt=dict(status='PASS',at=now(),token=token,fold=fold,history=history_id,
                                 natural_target=i in all_targets,benchmark_target=is_target,M0_choice=int(rec['choice']),
                                 candidates=count,valid=int(valid.sum()),native_candidate_calls=len(to_score),
                                 raw_log=log,current_index=i,source_sha256=sha(__file__),inference=rec,
                                 generation_seconds=generation_seconds,cache_seconds=cache_seconds,label_seconds=label_seconds,
                                 seconds=time.monotonic()-before,labels_traffic='reactive',proxy_EC_vs_actual_disagreements=int(np.sum(mask&np.isfinite(actual_ec)&np.isfinite(a['EC'])&(actual_ec!=a['EC']))),
                                 actual_EC_missing_on_valid=int(np.sum(valid&~np.isfinite(actual_ec))) if is_target else None)
                    write(dest/'receipt.json',receipt);timings.append(receipt)
                    if is_target:
                        completed.append(token)
                        print(json.dumps(dict(target_done=len(completed),token=token,valid=int(valid.sum()),generation_seconds=generation_seconds,cache_seconds=cache_seconds,label_seconds=label_seconds,wall_seconds=time.monotonic()-started)),flush=True)
                    signal.alarm(0)
                except Exception as exc:
                    signal.alarm(0)
                    failure=dict(at=now(),token=token,error=repr(exc),traceback=traceback.format_exc(),
                                 dependent_requested_tokens=[str(raw[j]['token'])for j in requested if j>=i], automatic_retry=False)
                    failures.append(failure);write(dest/'failure.json',failure)
                    write(RUN/'evidence/BENCHMARK_FAILURES.json',failures)
                    print(json.dumps(dict(failed_token=token,error=repr(exc))),flush=True)
                    break
    result=dict(status='PASS'if len(completed)==40 else'COMPLETED_WITH_FAILURES',at=now(),
                requested_targets=40,completed_targets=len(completed),completed_tokens=completed,
                failures=failures,inference_calls=inference_calls,native_calls=native_calls,
                load_seconds=load_seconds,wall_seconds=time.monotonic()-started,new_fits=0,
                timing_records=timings,IDM_compatibility_retries=idm_retries,
                outcome_aggregates_emitted=False)
    write(RUN/'evidence/BENCHMARK_RECEIPT.json',result)
    write(RUN/'QUEUE_STATUS.json',dict(at=now(),status='BENCHMARK_COMPLETE_AWAITING_ETA_DECISION',
          pid=os.getpid(),completed_targets=len(completed),target_total=40,failures=len(failures),new_fits=0))
    print(json.dumps({k:v for k,v in result.items()if k not in ['timing_records','failures','completed_tokens','IDM_compatibility_retries']}),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        write(RUN/'evidence/BENCHMARK_PROCESS_FAILURE.json',dict(at=now(),status='FAILED',error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False))
        write(RUN/'QUEUE_STATUS.json',dict(at=now(),status='FAILED',stage='benchmark',pid=os.getpid(),error=repr(exc),automatic_retry=False,new_fits=0))
        raise
