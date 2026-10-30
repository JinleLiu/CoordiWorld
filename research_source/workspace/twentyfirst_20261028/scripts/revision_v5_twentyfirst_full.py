"""Frozen QR/QR_local on complete historical DEVELOPMENT; reuse exact cache only.

Preparation uses current/history only. It freezes all selections before native
supplementation. The former R09 confirmation cohort is now EXPOSED DEVELOPMENT.
No candidate, reference, model, threshold or human-reference changes.
"""
from revision_v5_twentyfirst_common import *
import argparse,time,gc,copy,signal,traceback,shutil
import numpy as np,pandas as pd,xgboost as xgb
from revision_v5_twentieth_signal_features_fast import SignalGeometry,features,NAMES
from revision_v5_twentieth_route_correct import CachedRoute
from revision_v5_twentieth_route_signal_policy import apply_local_q
from revision_v5_eighteenth_projection import project
from revision_v5_eighteenth_policy import Policy
from revision_v5_seventeenth_progress import Extractor
from revision_v5_seventeenth_raw_history import raw_history_from_scene
from revision_v5_seventeenth_strict_aggregation import aggregate_complete,METRICS
from revision_v5_seventeenth_live import b
from navsim.common.dataclasses import Scene,SensorConfig,Trajectory
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

def inputs(scope,r):
    src=Path(r['source_folder'])if scope=='full140'else B19/'confirmation/scenes'/r['scene_id']
    a=dict(np.load(src/'inputs.npz',allow_pickle=False));assets=dict(np.load(src/'world_assets.npz',allow_pickle=False));current=json.load(open(src/'current.json'))
    if scope=='full140':old=dict(np.load(B18/'full_primary/scenes'/r['scene_id']/'predictions.npz',allow_pickle=False))
    else:old={k[len('prediction_'):]:v for k,v in a.items()if k.startswith('prediction_')};old.update({k:v for k,v in a.items()if k.startswith('extra_')})
    assert sha(src/'inputs.npz')==r['source_input_sha'if scope=='full140'else'input_sha']
    assert r['reference']==int(np.flatnonzero(a['valid'])[0]);assert current['latest_source_timestamp_us']<=current['timestamp_us']
    if scope=='exposed408':assert sha(src/'world_assets.npz')==r['world_sha']
    return src,a,assets,current,old

def history(r,logs):
    item=r['source'];path=item['server_path']
    if path not in logs['hashes']:assert sha(path)==item['sha256'];logs['hashes'].add(path)
    if r['origin']=='original':
        if logs.get('path')!=path:logs['frames']=b.load_log(path);logs['path']=path
        j=item['current_index'];h=logs['frames'][j-3:j+1]
    else:
        sc=Scene.load_from_disk(Path(path),None,SensorConfig.build_no_sensors());h=raw_history_from_scene(sc)
    assert len(h)==4 and h[-1]['token']==r['scene_id'];return h

def prepare(scope):
    oldroot,planpath=cohort(scope);out=BASE/scope;out.mkdir(parents=True,exist_ok=True)
    assert not(out/'selection_freeze.json').exists(),'preserve frozen selection'
    oldrows=json.load(open(oldroot/'choices.json'));plan=json.load(open(planpath));qpath=B20/'exploration/runs/QR/final/fit/model.ubj';assert sha(qpath)==Q_SHA
    model=xgb.Booster();model.load_model(qpath);assert model.num_features()==744
    policy=Policy({'value':'QV','nc':'NR','dac':'DR','projection':'IP','isotonic':False})
    oldbinding=json.load(open(B19/'PRIMARY_FREEZE_v02.json'))['head_bindings'];assert policy.bindings==oldbinding
    for h,r in oldbinding.items():assert sha(r['path'])==r['sha256']
    ext=SignalGeometry(ROOT/'data/maps');route=Extractor();logs={'hashes':set()};rows=[];t0=time.monotonic()
    write(out/'registration.json',dict(created_utc=now(),scope=scope,role='DEVELOPMENT; exposed historical population',plan_path=str(planpath),plan_sha=sha(planpath),source_choices_sha=sha(oldroot/'choices.json'),q_sha=Q_SHA,old_bindings=oldbinding,methods=METHODS,source_sha=sha(__file__),criteria_sha=sha(BASE/'MAIN_REGISTRATION.json')))
    for i,r in enumerate(oldrows):
        check_time();tt=time.monotonic();sid=r['scene_id'];folder=out/'scenes'/sid;folder.mkdir(parents=True,exist_ok=True)
        if(folder/'selection.json').exists():
            saved=json.load(open(folder/'selection.json'));assert saved['source_row_sha']==hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest();rows.append(saved);continue
        src,a,assets,current,old=inputs(scope,r)
        replay=policy.predict(a['x'],a['features198'][None],a['valid'][None],np.array([r['choices']['Geometry_HC']]),{'V24':old['extra_V24'],'R8':old['extra_R8']})
        for k in replay:np.testing.assert_array_equal(replay[k],old[k],err_msg='IC0 identity '+sid+' '+k)
        h=history(r,logs);sc=Scene.from_scene_dict_list(h,Path('.'),4,0,SensorConfig.build_no_sensors());scenario=NavSimScenario(sc,str(ROOT/'data/maps'),'nuplan-maps-v1.0');inp,init=route.processor._get_planner_inputs(scenario);planner=route.processor._pdm_closed;planner.initialize(init);planner._route_roadblock_correction(inp.history.current_state[0]);gc.enable();tokens=sorted(planner._route_lane_dict)
        if scope=='full140':
            g=json.load(open(oldroot/'scenes'/sid/'signal_geometry.json'));unrestricted=np.load(oldroot/'scenes'/sid/'predictions.npz')['signal_features']
        else:unrestricted,_,g=features(current,assets,a['candidates'],ext)
        f,detail,geom=features(current,assets,a['candidates'],CachedRoute(g,tokens));np.testing.assert_array_equal(f[:,:20],unrestricted[:,:20]);mask=a['valid']&f[:,[26,32,34]].astype(bool).any(-1);unmask=a['valid']&unrestricted[:,[26,32,34]].astype(bool).any(-1);assert not(mask&~unmask).any()
        xx=np.concatenate([a['x'][0],old['extra_V24'][0],f],-1).astype(np.float32);q=model.inplace_predict(xx);raw={k:old['raw_'+k][0]for k in ['q','qp','qttc','qlk']};raw['q']=q;proj=project(raw);score=(5*proj[:,1]+5*proj[:,2]+2*proj[:,3]+2*a['features198'][:,197]*proj[:,0])/14
        local=apply_local_q({k:v[0]for k,v in old.items()if k.startswith('raw_')or k=='eligible'},a['features198'],a['valid'],f,q)
        choices=dict(Geometry_HC=int(r['choices']['Geometry_HC']),IC0=int(old['choice'][0]),QR=int(np.where(old['eligible'][0],score,-np.inf).argmax()),QR_local=int(local['choice']))
        if not mask.any():assert choices['QR_local']==choices['IC0'];np.testing.assert_array_equal(local['scores'],old['scores'][0])
        if scope=='full140'and r['origin']=='original':assert not mask.any() and choices['QR_local']==choices['IC0']
        rec={k:r[k]for k in ['scene_id','drive','group','origin','source','reference']};rec.update(source_folder=str(src),source_row_sha=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest(),input_sha=sha(src/'inputs.npz'),world_sha=sha(src/'world_assets.npz'),current_sha=sha(src/'current.json'),candidate_sha=array_sha(a['candidates']),valid_sha=array_sha(a['valid']),choices=choices,mask_candidates=np.flatnonzero(mask).tolist(),unrestricted_mask_candidates=np.flatnonzero(unmask).tolist(),eligible_mask_candidates=np.flatnonzero(mask&old['eligible'][0]).tolist(),corrected_lane_ids=tokens,history_timestamps=[int(v['timestamp'])for v in h],candidates_to_evaluate=sorted(set(choices.values())),input_seconds=time.monotonic()-tt)
        np.savez_compressed(folder/'predictions.npz',route_features=f,unrestricted_features=unrestricted,x744=xx,candidates=a['candidates'],valid=a['valid'],features198=a['features198'],mask=mask,eligible=old['eligible'][0],QR_raw_q=q,QR_score=score,QR_projected=proj,QR_local_score=local['scores'],QR_local_raw_q=local['raw_q'],**{'IC0_'+k:old[k][0]for k in ['scores','raw_q','raw_qp','raw_qttc','raw_qlk','risk','dac']},**detail)
        write(folder/'route_geometry.json',geom);write(folder/'unrestricted_geometry.json',g);write(folder/'current.json',current);write(folder/'selection.json',rec);rows.append(rec)
        if(i+1)%20==0:print('R11_INPUT',scope,i+1,len(oldrows),round(time.monotonic()-t0,2),flush=True)
    assert len(rows)==len(oldrows) and len({r['scene_id']for r in rows})==len(rows)
    write(out/'choices.json',rows);bydrive={d:dict(scenes=sum(r['drive']==d for r in rows),support_scenes=sum(r['drive']==d and bool(r['mask_candidates'])for r in rows),support_candidates=sum(len(r['mask_candidates'])for r in rows if r['drive']==d),changed=sum(r['drive']==d and r['choices']['QR_local']!=r['choices']['IC0']for r in rows))for d in sorted({r['drive']for r in rows})}
    write(out/'support.json',dict(scenes=len(rows),mappings=len(plan['groups']),drives=len(bydrive),unrestricted_scenes=sum(bool(r['unrestricted_mask_candidates'])for r in rows),route_supported_scenes=sum(bool(r['mask_candidates'])for r in rows),supported_candidates=sum(len(r['mask_candidates'])for r in rows),supported_eligible_candidates=sum(len(r['eligible_mask_candidates'])for r in rows),original_supported=sum(r['origin']=='original'and bool(r['mask_candidates'])for r in rows),changed={m:sum(r['choices'][m]!=r['choices']['IC0']for r in rows)for m in METHODS},per_drive=bydrive,mask_subset_unrestricted=True,no_support_scores_choices_exact=True))
    write(out/'selection_freeze.json',dict(complete=True,created_utc=now(),choices_sha=sha(out/'choices.json'),support_sha=sha(out/'support.json'),registration_sha=sha(out/'registration.json'),q_model_sha=Q_SHA,new_native_results_read=False,seconds=time.monotonic()-t0));print('R11_FROZEN',scope,json.load(open(out/'support.json')),flush=True)

class Capture:
    def __init__(self,scorer):self.old=scorer.score_proposals;self.calls=[];scorer.score_proposals=self.score
    def score(self,*args,**kwargs):
        result=self.old(*args,**kwargs);i=1 if not self.calls else 0;self.calls.append(dict(role='candidate'if i==1 else'human',raw={k:float(result[i].iloc[0][k])for k in METRICS[:8]}));return result

def native(scope):
    oldroot,planpath=cohort(scope);out=BASE/scope;freeze=json.load(open(out/'selection_freeze.json'));assert freeze['complete']and sha(out/'choices.json')==freeze['choices_sha'];assert not(out/'result.json').exists()
    rows=json.load(open(out/'choices.json'));plan=json.load(open(planpath));t0=time.monotonic()
    from coordiworld.revision_v5.idm_teacher_geometry import install_corrected_builder
    install_corrected_builder()
    import nuplan.planning.simulation.observation.idm.idm_agent as idm
    from diagnose_idm_path import retry_dropped_endpoint
    from coordiworld.navsim_adapter import OfficialComponentScorer
    from navsim.evaluate.pdm_score import pdm_score
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses
    import revision_v5_collect_native_pilot as pilot
    oldpath=idm.create_path_from_se2;retries=[];idm.create_path_from_se2=lambda poses:retry_dropped_endpoint(oldpath,poses,retries,True);owner=OfficialComponentScorer(ROOT/'data/maps',out/'metric_cache','reactive');capture=Capture(owner.scorer)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('R11 native300s')))
    calls=0;reused=0;receipts=[]
    for i,r in enumerate(rows):
        check_time();folder=out/'scenes'/r['scene_id'];tt=time.monotonic();src=Path(r['source_folder']);a=np.load(src/'inputs.npz',allow_pickle=False);assert sha(src/'inputs.npz')==r['input_sha']and sha(src/'world_assets.npz')==r['world_sha'];assert array_sha(a['candidates'])==r['candidate_sha']and array_sha(a['valid'])==r['valid_sha'];assert int(a['valid'].argmax())==r['reference']
        if(folder/'native_receipt.json').exists():receipts.append(json.load(open(folder/'native_receipt.json')));continue
        signal.alarm(300);cache=None;human=None;checks=[]
        for cid in r['candidates_to_evaluate']:
            dest=folder/f'candidate_{cid}_filtered.pkl';pdest=folder/f'candidate_{cid}_physical.json';previous=oldroot/'scenes'/r['scene_id']/f'candidate_{cid}_{"filtered"if scope=="full140"else"raw"}.pkl';phys=oldroot/'scenes'/r['scene_id']/f'candidate_{cid}_physical.json'
            cachekey=dict(scene=r['scene_id'],candidate=cid,trajectory_sha=array_sha(a['candidates'][cid]),candidate_pool_sha=r['candidate_sha'],valid_sha=r['valid_sha'],reference=r['reference'],input_sha=r['input_sha'],world_sha=r['world_sha'],source_sha=r['source']['sha256'],protocol=PIN,traffic='reactive+declared_rearaxle_geometry+path_end_retry',human_reference='original fixed official human; synthetic absent',source_previous_sha=sha(previous)if previous.exists()else None)
            if previous.exists()and(scope=='full140'or r['origin']=='synthetic'):
                shutil.copy2(previous,dest);tab=pd.read_pickle(dest);row=tab.iloc[0]
                if phys.exists():shutil.copy2(phys,pdest)
                else:write(pdest,dict(candidate=cid,calls=[dict(role='candidate',raw={k:float(row[k])for k in METRICS[:8]})],filtered={k:float(row[k])for k in METRICS[:8]},physical_equals_filtered='synthetic official nohuman filtering',trajectory_sha=array_sha(a['candidates'][cid])))
                reused+=1;checks.append(dict(candidate=cid,reused=True,cache_key=cachekey));continue
            if cache is None:
                item=r['source'];assert sha(item['server_path'])==item['sha256']
                if r['origin']=='original':
                    frames=b.load_log(item['server_path']);j=item['current_index'];scene=Scene.from_scene_dict_list(frames[j-3:j+9],None,4,8,SensorConfig.build_no_sensors())
                else:scene=Scene.load_from_disk(Path(item['server_path']),None,SensorConfig.build_no_sensors())
                cache=owner.processor.compute_metric_cache(NavSimScenario(scene,str(ROOT/'data/maps'),'nuplan-maps-v1.0'))
            poses=a['candidates'][cid].astype(float);pilot.reset_rng(17);capture.calls=[];tab,states=pdm_score(copy.deepcopy(cache),Trajectory(poses,owner.prediction_sampling),owner.sampling,owner.simulator,owner.scorer,owner.traffic);captured=copy.deepcopy(capture.calls);assert len(captured)==(2 if r['origin']=='original'else 1)
            if len(captured)==2:
                if human is not None:assert human==captured[1]['raw']
                human=captured[1]['raw']
            ep=relative_to_absolute_poses(cache.ego_state.rear_axle,[StateSE2(*poses[-1])])[0];tab['valid']=True;tab['token']=r['scene_id'];tab['log_name']=cache.log_name;tab['frame_type']=cache.scene_type;tab['start_time']=cache.timepoint.time_s;tab['endpoint_x']=ep.x;tab['endpoint_y']=ep.y;tab['executed_endpoint_x']=states[-1,0];tab['executed_endpoint_y']=states[-1,1];tab['start_point_x']=cache.ego_state.rear_axle.x;tab['start_point_y']=cache.ego_state.rear_axle.y;tab['ego_simulated_states']=[states]
            err=None
            if previous.exists():
                prev=pd.read_pickle(previous);err=float(abs(prev[METRICS[:8]].to_numpy(float)-tab[METRICS[:8]].to_numpy(float)).max());assert err<1e-8;np.testing.assert_array_equal(prev.iloc[0].ego_simulated_states,states);np.testing.assert_array_equal(prev[['endpoint_x','endpoint_y']].to_numpy(),tab[['endpoint_x','endpoint_y']].to_numpy())
            tab.to_pickle(dest);write(pdest,dict(candidate=cid,calls=captured,filtered={k:float(tab.iloc[0][k])for k in METRICS[:8]},endpoint_source='official submitted lastpose/native initial rear axle',endpoint=[ep.x,ep.y],executed_endpoint=states[-1,:2].tolist(),trajectory_sha=array_sha(a['candidates'][cid])));calls+=1;checks.append(dict(candidate=cid,reused=False,old_components_maxerror=err,cache_key=cachekey))
        signal.alarm(0)
        if cache is not None:del cache,scene;gc.collect()
        rec=dict(scene_id=r['scene_id'],seconds=time.monotonic()-tt,checks=checks,human_fixed=True);write(folder/'native_receipt.json',rec);receipts.append(rec)
        if(i+1)%20==0:print('R11_NATIVE',scope,i+1,len(rows),calls,reused,round(time.monotonic()-t0,2),flush=True)
    mapping={(g['initial_token'],g['followup_token']):[tuple(p)for p in g['synthetic_initial_followup_pairs']]for g in plan['groups']};summary={};oldres=json.load(open(oldroot/'result.json'))
    for m in METHODS:
        raw=pd.concat([pd.read_pickle(out/'scenes'/r['scene_id']/f'candidate_{r["choices"][m]}_filtered.pkl')for r in rows],ignore_index=True);raw.to_pickle(out/(m+'-native.pkl'));scored,sm=aggregate_complete(raw,mapping,owner.sampling);scored.to_pickle(out/(m+'-scored.pkl'));summary[m]=sm
        if m in ['IC0','Geometry_HC']:assert sm==oldres['methods'][m],('cached baseline aggregate changed',m)
        for g in plan['groups']:
            mm={(g['initial_token'],g['followup_token']):[tuple(p)for p in g['synthetic_initial_followup_pairs']]};tokens={s for pair,pp in mm.items()for p in[pair,*pp]for s in p};_,ss=aggregate_complete(raw[raw.token.isin(tokens)].copy(),mm,owner.sampling);write(out/('group'+str(g['registered_group_index'])+'-'+m+'.json'),ss)
    write(out/'path_end_retries.json',retries);write(out/'native_cache_receipts.json',receipts);write(out/'result.json',dict(complete=True,created_utc=now(),scenes=len(rows),mappings=len(plan['groups']),drives=len({r['drive']for r in rows}),methods=summary,new_native_calls=calls,reused_candidate_labels=reused,seconds=time.monotonic()-t0,scope='exposed DEVELOPMENT; no independent confirmation',selection_freeze_sha=sha(out/'selection_freeze.json')));print('R11_FULL_RESULT',scope,{m:100*v['combined']['score']for m,v in summary.items()},flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','native']);ap.add_argument('scope',choices=['full140','exposed408']);a=ap.parse_args()
    try:(prepare if a.mode=='prepare'else native)(a.scope)
    except BaseException:write(BASE/a.scope/('failure_'+dt.datetime.now().strftime('%H%M%S')+'.json'),dict(created_utc=now(),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
