"""Post-freeze explanation only: supply, execution agreement and selected risk.
No training, calibration, thresholds, choices or cohort edits.
"""
from revision_v5_twentyfourth_common import *
from revision_v5_twentysecond_risk_quality import quality
from revision_v5_seventeenth_strict_aggregation import METRICS
import argparse,numpy as np,pandas as pd

def state_error(a,b):
    assert a.shape==b.shape
    return dict(position_m=float(np.linalg.norm(a[:,:2]-b[:,:2],axis=1).max()),yaw_rad=float(abs(np.arctan2(np.sin(a[:,2]-b[:,2]),np.cos(a[:,2]-b[:,2]))).max()),other_slots=float(abs(a[:,3:]-b[:,3:]).max()))

def main(scope):
    out=BASE/'validation'/scope;rows=json.load(open(out/'choices.json'));by={r['scene_id']:r for r in rows};contracts=json.load(open(out/'history_contracts.json'))
    native={m:pd.read_pickle(out/(m+'-native.pkl')).set_index('token')for m in METHODS};scored={m:pd.read_pickle(out/(m+'-scored.pkl')).set_index('token')for m in METHODS}
    records=[];base_rows=[]
    for r in rows:
        f=out/'base'/r['scene_id'];z=np.load(f/'features.npz');receipt=json.load(open(f/'receipt.json'))
        base_rows.append(dict(scene_id=r['scene_id'],drive=r['drive'],valid_candidates=int(z['valid'].sum()),brake=receipt['brake_supply'],yaw_drift=receipt['max_yaw_difference']))
    for r in contracts:
        method=r['method'];sid=r['current'];folder=out/'policy_inputs'/method/sid;z=np.load(folder/'features.npz');fm={int(k):v for k,v in r['feature_to_bank'].items()};k=next(k for k,v in fm.items()if v==r['choice']);pred=float(z['EC'][k]);actual=float(scored[method].loc[sid,'two_frame_extended_comfort'])
        errors=state_error(z['native'][k],native[method].loc[sid,'ego_simulated_states']);prior_errors=state_error(z['previous_native'],native[method].loc[r['previous'],'ego_simulated_states'])
        selected_variant=next((v['variant']for v in r['variants']if v.get('bank_index')==r['choice']and v['admitted']),'base_or_other_same_bytes')
        physical=json.load(open(out/'scenes'/sid/f'candidate_{r["choice"]}_physical.json'))['calls'][0]['raw']
        records.append(dict(scene_id=sid,drive=by[sid]['drive'],method=method,previous=r['previous'],previous_choice=r['previous_choice'],choice=r['choice'],feature_index=k,trigger=r['trigger'],attempted=len(r['variants']),admitted=sum(v['admitted']for v in r['variants']),selected_new=k>=12,selected_variant=selected_variant,predicted_EC=pred,actual_EC=actual,EC_mismatch=pred!=actual,
            world_recompute_maxerr=max([v.get('world_maxerr',0.)for v in r['variants']]+[0.]),**{'current_'+x:y for x,y in errors.items()},**{'previous_'+x:y for x,y in prior_errors.items()},
            **{'raw_'+c:float(physical[c])for c in METRICS[:8]},**{'filtered_'+c:float(scored[method].loc[sid,c])for c in METRICS[:8]}))
    df=pd.DataFrame(records);df.to_csv(out/'supply_and_execution.csv',index=False);bd=pd.DataFrame(base_rows);bd.to_csv(out/'candidate_coverage.csv',index=False);summary={}
    for method in METHODS:
        part=df[df.method==method];summary[method]=dict(current_pairs=len(part),triggered=int(part.trigger.sum()),attempted=int(part.attempted.sum()),admitted=int(part.admitted.sum()),selected_new=int(part.selected_new.sum()),selected_sources=part[part.selected_new].selected_variant.value_counts().to_dict(),EC_mismatches=part[part.EC_mismatch].scene_id.tolist(),
            max_current_position_error=float(part.current_position_m.max()),max_previous_position_error=float(part.previous_position_m.max()),max_yaw_error=float(max(part.current_yaw_rad.max(),part.previous_yaw_rad.max())),max_other_state_slot_error=float(max(part.current_other_slots.max(),part.previous_other_slots.max())),max_world_recompute_error=float(part.world_recompute_maxerr.max()))
    # Probability comparison on the one already frozen primary population.
    # Constant rate is inherited from TRAIN, never fitted on this scope.
    old=ROOT/'twentythird_20261029/fixed_population_risk/result.json';rate=json.load(open(old))['constant']
    saved=json.load(open(BASE/'delivery/package_data/data'/scope/'selection_replay.json'));prob_rows=[]
    for r in saved:
        if r['method']!='PNC_H3P':continue
        sid=r['scene_id'];cid=r['expected_bank_choice'];k=next(int(k)for k,v in r['feature_to_bank'].items()if v==cid)
        pred=float(r['predictions']['raw_physical_nc']['values'][k]);y=float(json.load(open(out/'scenes'/sid/f'candidate_{cid}_physical.json'))['calls'][0]['raw'][METRICS[0]]<1)
        prob_rows.append(dict(scene_id=sid,drive=by[sid]['drive'],candidate=cid,new_candidate=k>=12,event=y,probability=pred,TRAIN_constant=rate))
    pr=pd.DataFrame(prob_rows);pr.to_csv(out/'fixed_primary_risk.csv',index=False);d=pr.drive.to_numpy();weights=np.array([1/(d==x).sum()for x in d]);y=pr.event.to_numpy();p=pr.probability.to_numpy()
    risk={name:quality(value,y,weights)for name,value in [('raw_NC_BCE',p),('TRAIN_constant',np.full(len(y),rate)),('constant_zero',np.zeros(len(y)))]}
    delta=(p-y)**2-(rate-y)**2;drives=sorted(set(d));dv=np.array([delta[d==x].mean()for x in drives]);rng=np.random.default_rng(102819);ci=np.quantile(dv[rng.integers(len(dv),size=(10000,len(dv)))].mean(1),[.025,.975]);risk['paired_vs_TRAIN_constant']=dict(delta_Brier=float(dv.mean()),CI95_drive_cluster=ci.tolist())
    for name,mask in [('new_candidate',pr.new_candidate.to_numpy()),('base_candidate',~pr.new_candidate.to_numpy())]:
        risk[name]={n:quality(v[mask],y[mask],weights[mask])for n,v in [('raw_NC_BCE',p),('TRAIN_constant',np.full(len(y),rate))]}if mask.any()else None
    coverage=dict(scenes=len(bd),drives=len(set(bd.drive)),valid_count_histogram={str(k):int(v)for k,v in bd.valid_candidates.value_counts().sort_index().items()},multi_candidate_scenes=int((bd.valid_candidates>1).sum()),multi_candidate_drives=len(set(bd[bd.valid_candidates>1].drive)),brake_scenes=int(bd.brake.sum()),max_yaw_drift=float(bd.yaw_drift.max()),p95_yaw_drift=float(bd.yaw_drift.quantile(.95)))
    result=dict(complete=True,created_utc=now(),scope=scope,selection_frozen=True,method_changes=False,coverage=coverage,supply=summary,risk=risk,TRAIN_constant=rate,TRAIN_constant_source=str(old),TRAIN_constant_source_sha=sha(old),risk_scope='ONE fixed primary selected population, drive equal then scene equal; raw NC<1 includes0.5; positive/negative decomposition. No calibration or selection update; not real collision probability.',EC_note='prediction/actual comparison uses native own execution and timestamp alignment; stage score impact and endpoint effects remain distinct')
    write(out/'mechanism.json',result);print('R14_MECHANISM',scope,coverage,{m:(s['selected_new'],len(s['EC_mismatches']))for m,s in summary.items()},'RISK',risk['raw_NC_BCE']['Brier'],risk['TRAIN_constant']['Brier'],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('scope');main(ap.parse_args().scope)
