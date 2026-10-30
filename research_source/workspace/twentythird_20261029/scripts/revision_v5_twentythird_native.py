"""R12 full protocol using unchanged native candidate cache and own EC/weights."""
from revision_v5_twentythird_common import *
import argparse,time,gc,copy,signal,traceback,shutil
import numpy as np,pandas as pd,xgboost as xgb
from revision_v5_twentysecond_train import inputs,decide,HEADS
from revision_v5_seventeenth_strict_aggregation import aggregate_complete,METRICS
from revision_v5_seventeenth_live import b
from navsim.common.dataclasses import Scene,SensorConfig,Trajectory
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
STUDY=B22
BASE=BASE/'validation/continuation'
METHODS=['Geometry_HC','IC0','PNC_EC','Geometry_pool_EC','IC0_pool_EC','PNC_CONT','IC0_CONT','Geometry_CONT']
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def cohort(scope):return B21/scope,B21/'replay'/(scope+'_plan.json')
def prepare(scope):
 out=BASE/scope;out.mkdir(parents=True,exist_ok=True);assert not(out/'selection_freeze.json').exists();a=dict(np.load(STUDY/'features'/scope/'dataset.npz'));xx=inputs(a);rows=json.load(open(B21/scope/'choices.json'));models={};bindings={}
 for h in HEADS:
  path=STUDY/'runs/K0/final/fit'/h/'model.ubj';m=xgb.Booster();m.load_model(path);models[h]=m;bindings[h]=dict(path=str(path),sha256=sha(path))
 raw={h:m.inplace_predict(xx[h].reshape(-1,xx[h].shape[-1])).reshape(len(rows),12)for h,m in models.items()};p=decide(raw,a,np.arange(len(rows)));result=[]
 for i,r in enumerate(rows):
  assert r['scene_id']==a['scene_ids'][i];r=copy.deepcopy(r);r['choices']={k:r['choices'][k]for k in ['Geometry_HC','IC0','QR_local']};r['choices'].update(K0=int(p['choice'][i]),K0_QR_local=int(p['QR_local_choice'][i]));r['candidates_to_evaluate']=sorted(set(r['choices'].values()));folder=out/'scenes'/r['scene_id'];folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'predictions.npz',**{k:v[i]for k,v in p.items()});write(folder/'selection.json',r);result.append(r)
 write(out/'choices.json',result);write(out/'registration.json',dict(created_utc=now(),methods=METHODS,scope='exposed DEVELOPMENT',input_sha=sha(STUDY/'features'/scope/'dataset.npz'),models=bindings,source_sha=sha(__file__),protocol=PIN,endpoint='submitted last pose transformed by initial rear axle',EC='actual own adjacent execution states'))
 write(out/'selection_freeze.json',dict(complete=True,created_utc=now(),choices_sha=sha(out/'choices.json'),registration_sha=sha(out/'registration.json'),new_native_results_read=False));print('B22_FULL_FROZEN',scope,{m:sum(r['choices'][m]!=r['choices']['IC0']for r in result)for m in METHODS},flush=True)
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
            dest=folder/f'candidate_{cid}_filtered.pkl';pdest=folder/f'candidate_{cid}_physical.json';oldcid=r.get('evaluation_to_old',{}).get(str(cid),cid) if cid!=r.get('new_candidate_index',-1) else -1;previous=oldroot/'scenes'/r['scene_id']/f'candidate_{oldcid}_filtered.pkl';phys=oldroot/'scenes'/r['scene_id']/f'candidate_{oldcid}_physical.json'
            cachekey=dict(scene=r['scene_id'],candidate=cid,trajectory_sha=array_sha(a['candidates'][cid]),candidate_pool_sha=r['candidate_sha'],valid_sha=r['valid_sha'],reference=r['reference'],input_sha=r['input_sha'],world_sha=r['world_sha'],source_sha=r['source']['sha256'],protocol=PIN,traffic='reactive+declared_rearaxle_geometry+path_end_retry',human_reference='original fixed official human; synthetic absent',source_previous_sha=sha(previous)if previous.exists()else None)
            if not previous.exists() and oldcid>=0:
                for cache_root in [ROOT/'twentysecond_20261028/critical_pools',ROOT/'twentysecond_20261028/validation/coordinate'/scope/'scenes']:
                    alt=cache_root/r['scene_id']/f'candidate_{oldcid}_filtered.pkl'
                    if alt.exists():previous=alt;phys=alt.with_name(f'candidate_{oldcid}_physical.json');break
            samepool_used=False
            pcid=r.get('physical_cache_remap',{}).get(str(cid),cid if 'smooth_slot' not in r else -1)
            if pcid>=0:
                for tag in ['shared_ec','physical_risk','supply']:
                    alt=STUDY/'validation'/tag/scope/'scenes'/r['scene_id']/f'candidate_{pcid}_filtered.pkl'
                    if alt.exists():
                        sr=json.load(open(alt.parent/'selection.json'));oldinput=np.load(Path(sr['source_folder'])/'inputs.npz');np.testing.assert_array_equal(a['candidates'][cid],oldinput['candidates'][pcid]);previous=alt;phys=alt.with_name(f'candidate_{pcid}_physical.json');samepool_used=True;break
            else:previous=out/'never_reuse_new_slot';phys=previous
            if previous.exists():
                if not samepool_used:
                    oldinput=np.load(Path(r.get('old_source_folder',r['source_folder']))/'inputs.npz');np.testing.assert_array_equal(a['candidates'][cid],oldinput['candidates'][oldcid])
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
 global BASE
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','native']);ap.add_argument('scope',choices=['full140','exposed408']);ap.add_argument('--tag',default='coordinate');a=ap.parse_args();BASE=STUDY/'validation'/a.tag
 try:(prepare if a.mode=='prepare'else native)(a.scope)
 except BaseException:write(BASE/a.scope/('failure_'+dt.datetime.now().strftime('%H%M%S')+'.json'),dict(created_utc=now(),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
