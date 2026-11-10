"""Original 2919-input/1336 roster, input-only capacity and inherited fallback audit."""
from coordiworld_publication import publication_identifier
import json,pickle,fcntl,time,datetime,sys
from pathlib import Path
import align_train_benchmark as b
from a0_causal_context_manifest import history_kind
import numpy as np


def main():
    held=(b.RUN/'locks/OWN_HISTORY_INPUT_AUDIT_RUN.lock').open('a+');fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();r18=b.R18;blocks=[];flat=[];pins={}
    for path in sorted((r18/'data').glob('cohort_block[0-9][0-9][0-9].json')):
        rows=json.loads(path.read_text())['records'];pins[str(path)]=b.sha(path)
        block=dict(block=int(path.stem[-3:]),metadata_path=str(path),metadata_sha256=b.sha(path),records=rows)
        blocks.append(block);flat.extend(rows)
    assert len(flat)==2919 and len({x['token'] for x in flat})==2919 and sum(bool(x['scored']) for x in flat)==1336
    assert len({x['drive'] for x in flat if x['scored']})==20
    cfg=b.pnc.m.yaml.safe_load((b.pnc.m.PEER/'config/group_cache_v2.yaml').read_text());capacity=300
    known=json.loads((b.RUN/'configs/INPUT_CAPACITY_MISSINGNESS_v01.json').read_text())
    expected=json.loads((r18/'code/pnc_expected_bindings.json').read_text())
    assert known['checkpoint_sha256']==expected['world'] and known['model_capacity']==capacity
    model=b.pnc.FrozenPNC();assert {k:v['sha256'] for k,v in model.bindings.items()}==expected['heads']
    bylog={}
    for row in flat:bylog.setdefault(row['raw_log'],[]).append(row)
    details={};fallbacks=[];cache_missing=[]
    for path,rows in bylog.items():
        blob=Path(path).read_bytes();digest=b.sha(path);assert all(x['raw_log_sha256']==digest for x in rows);raw=pickle.loads(blob)
        for row in rows:
            i=row['index'];token=row['token'];frame=raw[i];assert str(frame['token'])==token
            assert i>=3 and len(raw[i-3:i+1])==4
            kind=history_kind(raw,i);cur=b.pnc.m.frame_agents(frame,np.asarray(b.pnc.m.ego_pose(frame)))
            count=int(max(1,sum(np.linalg.norm(s[:2])<=cfg['data']['roi_m'] for s,_ in cur.values())))
            prior_path=r18/'predictions/CoordiWorld-PNC_H3P'/token/'receipt.json';prior=json.loads(prior_path.read_text())
            fallback=bool(prior.get('input_contract_fallback',False))
            if fallback:
                assert token==publication_identifier('source_scene_006') and not row['scored']
                old_binding=r18/'analysis/pnc_input_fallback_binding_v01.json';binding=json.loads(old_binding.read_text())
                base=r18/'results/PDM-Closed'/token
                assert b.sha(base/'trajectory.npy')==binding['trajectory_sha256'] and b.sha(base/'execution.npy')==binding['execution_sha256']
                fallbacks.append(dict(token=token,reason=prior.get('fallback_reason'),binding_path=str(old_binding),binding_sha256=b.sha(old_binding),
                    trajectory_path=str(base/'trajectory.npy'),trajectory_sha256=binding['trajectory_sha256'],
                    execution_path=str(base/'execution.npy'),execution_sha256=binding['execution_sha256'],
                    inherited=True,scored=False,readout_available=False))
            elif kind not in ['original_regular_four','inherited_R18_causal_oldest_slot_adapter']:
                raise RuntimeError(('New unsupported benchmark input; do not invent fallback',token,kind))
            cache=r18/'data/metric_cache'/row['log']/'unknown'/token/'metric_cache.pkl'
            if not cache.exists():cache_missing.append(token)
            details[token]=dict(input_kind=kind,current_actor_slots=count,capacity=capacity,direct_capacity_overflow=count>capacity,
                inherited_fallback=fallback,current_timestamp_us=int(frame['timestamp']),prior_receipt_sha256=b.sha(prior_path),
                metric_cache_path=str(cache),metric_cache_exists=cache.exists())
    assert len(fallbacks)==1
    blocked=[];changes=[]
    for block in blocks:
        prev=None;blocker=None
        for j,row in enumerate(block['records']):
            d=details[row['token']];ts=d['current_timestamp_us']
            reset=j==0 or row.get('segment_start',False) or (prev is not None and not 0<ts-prev<550000)
            if reset:blocker=None
            if d['direct_capacity_overflow'] and not d['inherited_fallback'] and blocker is None:blocker=row['token']
            row['input_contract']=d|dict(causal_reset_before=bool(reset),causal_blocker=blocker)
            if blocker:blocked.append(dict(token=row['token'],scored=bool(row['scored']),root=blocker,block=block['block']))
            prev=ts
    bins=[[],[],[]];costs=[0,0,0]
    for block in sorted(blocks,key=lambda x:(-len(x['records']),x['block'])):
        j=min(range(3),key=lambda k:(costs[k],k));bins[j].append(block['block']);costs[j]+=len(block['records'])
    at=b.now();manifest=dict(at=at,status='INPUT_AUDITED_BEFORE_NEW_OWN_HISTORY',source_sha256=b.sha(__file__),
        original_T0='2026-11-02T14:47:14.687+08:00',strategy='anchored_P',formal_scenes=1336,input_nodes=2919,drives=20,
        metadata_pins=pins,blocks=blocks,worker_blocks=[sorted(x) for x in bins],worker_counts=costs,workers=3,
        production_M0_head_bindings=model.bindings,world_sha256=expected['world'],capacity=capacity,
        inherited_fallbacks=fallbacks,capacity_blocked=blocked,missing_metric_cache=cache_missing,
        history='Own P selected predicted controller execution; recompute H3 candidates and EC in sequence. Original block/segment/timestamp resets only. No reset across new structural unavailability.',
        labels='Official non-reactive human-filtered native scorer plus actual own adjacent EC; original1336 denominator. No outcome may affect selection or new strategy.',
        data_exposure='R18 previously exposed20drive benchmark; not independent confirmation',
        old_deadline_used=False,selection_outcomes_read=False,model_forwards=0,native_scores=0,new_fits=0)
    dest=b.RUN/'configs/P_OWN_HISTORY_INPUT_MANIFEST_v01.json';assert not dest.exists();b.write(dest,manifest)
    result=dict(at=at,status='PASS' if not cache_missing and not blocked else 'REQUIRES_EXPLICIT_MISSINGNESS_HANDLING',
        manifest_sha256=b.sha(dest),metadata_files=len(blocks),input_nodes=2919,formal_scenes=1336,drives=20,
        max_actor_slots=max(x['current_actor_slots'] for x in details.values()),capacity=capacity,
        inherited_fallbacks=len(fallbacks),capacity_blocked=blocked,missing_metric_cache=cache_missing,
        input_kinds={k:sum(x['input_kind']==k for x in details.values()) for k in sorted({x['input_kind'] for x in details.values()})},
        worker_counts=costs,seconds=time.monotonic()-start,new1336_predictions_or_outcomes_read=False)
    b.write(b.RUN/'evidence/P_OWN_HISTORY_INPUT_AUDIT_v01.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
