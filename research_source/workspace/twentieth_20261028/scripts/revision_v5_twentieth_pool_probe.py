"""TRAIN post-freeze M12 brake-coverage probe. New slots never inherit old labels.

Primary metadata cohort and two outcome-defined diagnostic cases stay separate.
Reference, world, controller, 12 slots and all unaffected candidate bytes fixed.
This is frozen-head transfer, not a matched-retrained new-pool performance claim.
"""
import os
os.environ.setdefault('NUPLAN_MAPS_ROOT',__import__('coordiworld_paths').workspace_path('data/maps'))
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,time,copy,hashlib,gc,signal,traceback
from pathlib import Path
import numpy as np,torch,yaml,xgboost as xgb
from revision_v5_twentieth_q_train import ROOT,BASE,OLD,write,sha,now
from revision_v5_seventeenth_live import Pipeline,b
from revision_v5_eighteenth_policy import Policy
from revision_v5_eighteenth_source_features import features as vf
from revision_v5_eighteenth_road import features as rf
from revision_v5_eighteenth_projection import project
from revision_v5_twentieth_signal_features import SignalGeometry,features
from revision_v5_twentieth_full import Capture,METRICS
from coordiworld.revision_v5.native_adapter import read_legacy_inputs
import revision_v5_collect_native_pilot as pilot

def braking(reference,speed):
    # Follow reference spatial path; monotone arc interpolation, no extrapolation.
    pp=np.vstack([np.zeros(3),reference]).astype(float);arc=np.r_[0.,np.linalg.norm(np.diff(pp[:,:2],axis=0),axis=-1).cumsum()];keep=np.r_[True,np.diff(arc)>1e-7];pp=pp[keep];arc=arc[keep];t=np.arange(1,9)*.5;tt=np.minimum(t,speed/6);dist=speed*tt-3*tt**2;query=np.minimum(dist,arc[-1]);pp[:,2]=np.unwrap(pp[:,2]);poses=np.stack([np.interp(query,arc,pp[:,j])for j in range(3)],-1).astype(np.float32);full=np.vstack([np.zeros(3),poses]);sp=np.linalg.norm(np.diff(full[:,:2],axis=0),axis=-1)/.5;acc=np.diff(np.r_[speed,sp])/.5;yr=np.diff(np.unwrap(full[:,2]))/.5;ok=bool(acc.min()>=-6.15 and acc.max()<=3.15 and abs(yr).max()<=.55 and (sp*abs(yr)).max()<=4.2)
    return poses,dict(immediate_longitudinal_deceleration_mps2=6.,reference_arc_truncated=bool((dist>arc[-1]).any()),discrete_admissible=ok,min_discrete_accel=float(acc.min()),max_lateral_accel=float((sp*abs(yr)).max()),max_yaw_rate=float(abs(yr).max()))
def slot(c,v,ref):
    invalid=[j for j in range(ref+1,12)if not v[j]]
    if invalid:return max(invalid),'previously_invalid'
    pairs=[]
    for j in range(12):
        if j==ref or not v[j]:continue
        d=min(np.linalg.norm(c[j,:,:2]-c[k,:,:2],axis=-1).max()for k in range(12)if k!=j and v[k]);pairs.append((d,-j,j))
    assert pairs,'No reference-preserving replacement slot'
    return min(pairs)[2],'closest_to_retained_candidate_by_max_XY_distance'
def main():
    out=BASE/'exploration/brake_pool_v02';out.mkdir(parents=True,exist_ok=False);z=np.load(ROOT/'sixteenth_20261027/train_inputs.npz');sf=np.load(BASE/'signal_train/features.npz')['features'];inv={r['scene_id']:r for r in json.load(open(ROOT/'runs/paired_geometry_v2_train128_dev64/input_screening.json'))['rows']};valid=z['valid'];condition=(sf[:,:,11]*20>3)&((sf[:,:,34]>.5)|(sf[:,:,26]>.5))&valid;inds=np.flatnonzero(condition.any(1));selected=[];drives=set()
    for i in sorted(inds,key=lambda i:hashlib.sha256(('B20_BRAKE_SUPPORT|'+str(z['scene_ids'][i])).encode()).hexdigest()):
        if str(z['drives'][i])in drives:continue
        selected.append(int(i));drives.add(str(z['drives'][i]))
        if len(selected)==12:break
    cohort={i:'current_condition_metadata12'for i in selected}
    for sid in ['study_scene_001140','study_scene_002495']:
        i=int(np.flatnonzero(z['scene_ids']==sid)[0]);cohort.setdefault(i,'outcome_defined_all_TLC_bad_diagnostic')
    selected=list(cohort);write(out/'registration.json',dict(created_utc=now(),bottleneck='Two TRAIN all-TLC-bad scenes; most others have safe old candidates. Unknown/future signal does not authorize red hardfilter.',mechanism='Add one immediate6mps2 route-following brake while retaining original reference. Replace invalid or most spatially redundant nonreference slot, same12slots; actual controller may overshoot.',minimal_control='old/newM12 x frozen Geometry/IC0, plus QS exploratory transfer. No new-pool retraining; only selected14TRAIN. New candidate native labels separately hashed.',continue_if='New brake increases available safety/utility without losing unique old coverage, and current-input selection can use it; if coverage improves but selection fails, distinguish bottlenecks before scaling',main_model_unchanged=True,primary_ids=[str(z['scene_ids'][i])for i in selected if cohort[i]=='current_condition_metadata12'],diagnostic_ids=[str(z['scene_ids'][i])for i in selected if cohort[i]!='current_condition_metadata12'],source_sha=sha(__file__)))
    pipe=Pipeline(yaml.safe_load(open(ROOT/'runtime_s1/configs/group_cache_v2.yaml')));pol=Policy(dict(value='QV',nc='NR',dac='DR',projection='IP',isotonic=False));qm=xgb.Booster();qm.load_model(BASE/'runs/QS/final/fit/model.ubj');ext=SignalGeometry(ROOT/'data/maps');recs=[];start=time.monotonic()
    for i in selected:
        sid=str(z['scene_ids'][i]);r=inv[sid];path=ROOT/'data/openscene/trainval'/(r['log_name']+'.pkl');assert sha(path)==r['raw_log_sha256'];frames=b.load_log(path);history=frames[r['index']-3:r['index']+1];scene,meta=read_legacy_inputs(r['cache_path'],r);orig=scene.candidates[0].numpy().copy();v=scene.candidate_valid[0].numpy().copy();ref=int(v.argmax());j,reason=slot(orig,v,ref);new,br=braking(orig[ref],np.linalg.norm(history[-1]['ego_dynamic_state'][:2]));duplicate=bool(any(np.linalg.norm(new[:,:2]-orig[k,:,:2],axis=-1).max()<.001 for k in np.flatnonzero(v)if k!=j));applied=br['discrete_admissible']and not duplicate;dest=out/'scenes'/sid;dest.mkdir(parents=True);choices={};preds={}
        for kind in ['old','new']:
            c=orig.copy();vv=v.copy()
            if kind=='new'and applied:c[j]=new;vv[j]=True
            scene.candidates=torch.from_numpy(c[None]);scene.candidate_valid=torch.from_numpy(vv[None]);a=pipe.inputs(history,r['raw_log_sha256'],scene,meta);extra={'V24':vf(a['assets'])[0][None],'R8':rf(a['areas'],a['valid'])[None]};p=pol.predict(a['x'],a['features198'][None],a['valid'][None],np.array([a['Geometry_HC']]),extra);sig,_,_=features(a['current'],a['assets'],c,ext);xx=np.concatenate([a['x'][0],extra['V24'][0],sig],-1).astype(np.float32);q=qm.inplace_predict(xx);raw={h:p['raw_'+h][0]for h in ['q','qp','qttc','qlk']};raw['q']=q;pr=project(raw);score=(5*pr[:,1]+5*pr[:,2]+2*pr[:,3]+2*a['features198'][:,197]*pr[:,0])/14;qs=int(np.where(p['eligible'][0],score,-np.inf).argmax());choices[kind]=dict(Geometry_HC=a['Geometry_HC'],IC0=int(p['choice'][0]),QS=qs);preds[kind]=dict(a=a,c=c,valid=vv)
            np.savez_compressed(dest/(kind+'_inputs.npz'),candidates=c,valid=vv,features198=a['features198'],x=a['x'],signal=sig,world=a['assets']['world'],execution=a['assets']['execution_full_local'],IC0_choice=p['choice'],IC0_scores=p['scores'],eligible=p['eligible'],QS_q=q,QS_score=score)
        np.testing.assert_array_equal(preds['old']['c'][ref],preds['new']['c'][ref]);np.testing.assert_array_equal(preds['old']['a']['assets']['world'],preds['new']['a']['assets']['world']);legacy_error=float(np.max(np.abs(preds['old']['a']['features198']-z['features'][i])));assert legacy_error<1e-3,legacy_error
        unchanged=np.arange(12)!=j;np.testing.assert_array_equal(preds['old']['c'][unchanged],preds['new']['c'][unchanged]);np.testing.assert_array_equal(preds['old']['a']['assets']['execution_full_local'][unchanged],preds['new']['a']['assets']['execution_full_local'][unchanged]);rec=dict(scene_id=sid,index=i,drive=str(z['drives'][i]),cohort=cohort[i],source=r,reference=ref,replaced_slot=j,slot_rule=reason,old_slot_valid=bool(v[j]),applied=applied,duplicate=duplicate,brake=br,choices=choices,legacy_f198_max_abs_difference=legacy_error,baseline='fresh current pipeline old pool paired with same fresh pipeline new pool; tiny historical geometry float drift saved',reference_world_exact=True,other_candidate_execution_exact=True,new_trajectory_sha=hashlib.sha256(new.tobytes()).hexdigest());write(dest/'selection.json',rec);recs.append(rec);print('B20_BRAKE_FROZEN',sid,applied,choices,flush=True)
    write(out/'selection_freeze.json',dict(created_utc=now(),records=recs,all_choices_before_new_labels=True));b.OUT=out;b.initialize_scoring();owner=b.SCORER;cap=Capture(owner.scorer);results=[]
    for r in recs:
        sid=r['scene_id'];i=r['index'];src=r['source'];frames=b.load_log(ROOT/'data/openscene/trainval'/(src['log_name']+'.pkl'));cache=owner.build_metric_cache(frames,src['index']);a=np.load(out/'scenes'/sid/'new_inputs.npz');c=a['candidates'];ref=r['reference'];j=r['replaced_slot'];label={};calls={}
        for k in sorted(set([ref,j]if r['applied']else[ref])):
            cap.calls=[];pilot.reset_rng(17);label[str(k)]=owner.score(copy.deepcopy(cache),c[k]);calls[str(k)]=copy.deepcopy(cap.calls)
        comp=np.array([[label[str(ref)][k]for k in METRICS[:8]]]);np.testing.assert_allclose(comp[0],z['components'][i,ref],atol=1e-7,rtol=0);cc=z['components'][i].copy();vv=z['valid'][i].copy()
        if r['applied']:cc[j]=[label[str(j)][k]for k in METRICS[:8]];vv[j]=True
        def utility(c):return c[...,:4].prod(-1)*(5*c[...,4]+5*c[...,5]+2*c[...,6]+2*c[...,7])/14
        row=dict(scene_id=sid,cohort=r['cohort'],new_scores=label,calls=calls,reference_match=True,old_oracle_U=float(z['utility'][i,vv if not r['applied']else z['valid'][i]].max()),new_oracle_U=float(utility(cc)[vv].max()),old_NC_safe=int((z['components'][i,:,0][z['valid'][i]]==1).sum()),new_NC_safe=int((cc[vv,0]==1).sum()),old_TLC_safe=int((z['components'][i,:,3][z['valid'][i]]==1).sum()),new_TLC_safe=int((cc[vv,3]==1).sum()))
        for kind,ccc in [('old',z['components'][i]),('new',cc)]:
            for m,k in r['choices'][kind].items():row[kind+'_'+m]=dict(candidate=k,components=ccc[k].tolist(),U=float(utility(ccc[k])))
        results.append(row);write(out/'scenes'/sid/'native.json',row);print('B20_BRAKE_NATIVE',sid,row['old_oracle_U'],row['new_oracle_U'],row['old_TLC_safe'],row['new_TLC_safe'],flush=True)
    write(out/'result.json',dict(complete=True,seconds=time.monotonic()-start,scenes=len(recs),records=results,main_unchanged=True,scope='14TRAIN mechanism probe, frozen final heads are in-sample; no OOF/generalization claim; primary12 and2outcome-defined diagnostics separate'))
if __name__=='__main__':
    try:main()
    except BaseException:write(BASE/'exploration/brake_pool_v02/failure.json',dict(traceback=traceback.format_exc()));raise
