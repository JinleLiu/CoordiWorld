"""Frozen QS/QE DEVELOPMENT full140 with separate physical and human-filtered metrics.

No training or recipe selection here. All selections freeze before native outcomes.
Official submitted terminal XY sets weights; own actual execution sets two-frame EC.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import copy,json,time,signal,gc,traceback
from pathlib import Path
import numpy as np,pandas as pd,xgboost as xgb
from revision_v5_twentieth_q_train import ROOT,BASE,OLD,write,sha,now
from revision_v5_twentieth_signal_features import SignalGeometry,features,NAMES
from revision_v5_eighteenth_source_features import features as v24
from revision_v5_eighteenth_projection import project
from revision_v5_seventeenth_strict_aggregation import aggregate_complete,METRICS
from revision_v5_seventeenth_live import b
from navsim.common.dataclasses import Scene,SensorConfig,Trajectory
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.evaluate.pdm_score import pdm_score
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
import revision_v5_collect_native_pilot as pilot
METHODS=['Geometry_HC','IC0','QE','QS']
class Capture:
    def __init__(self,scorer):self.old=scorer.score_proposals;self.calls=[];scorer.score_proposals=self.score
    def score(self,*args,**kwargs):
        result=self.old(*args,**kwargs);i=1 if not self.calls else 0
        self.calls.append(dict(role='candidate'if i==1 else'human',raw={k:float(result[i].iloc[0][k])for k in METRICS[:8]}))
        return result

def main():
    freeze=json.load(open(BASE/'MAIN_FREEZE.json'));assert freeze['selected_recipe']=='QS'
    out=BASE/'development_full';out.mkdir(exist_ok=True);t0=time.monotonic()
    assert not(out/'result.json').exists(),'preserve completed run'
    plan=json.load(open(OLD/'FULL_INPUT_LOCAL.json'));oldrows=json.load(open(OLD/'full_primary/choices.json'));ext=SignalGeometry(ROOT/'data/maps');models={};bindings={}
    for name in ['QE','QS']:
        path=BASE/'runs'/name/'final/fit/model.ubj';model=xgb.Booster();model.load_model(path);models[name]=model;bindings[name]=dict(path=str(path),sha256=sha(path))
    rows=[]
    if (out/'selection_freeze.json').exists():
        rows=json.load(open(out/'choices.json'));assert json.load(open(out/'selection_freeze.json'))['model_bindings']==bindings
    else:
        for n,r in enumerate(oldrows):
            sid=r['scene_id'];src=Path(r['source_folder']);a=dict(np.load(src/'inputs.npz'));assets=dict(np.load(src/'world_assets.npz'));current=json.load(open(src/'current.json'));old=dict(np.load(OLD/'full_primary/scenes'/sid/'predictions.npz'));folder=out/'scenes'/sid;folder.mkdir(parents=True,exist_ok=True)
            sf,detail,geometry=features(current,assets,a['candidates'],ext);extra=v24(assets)[0];base=np.concatenate([a['x'][0],extra],-1);cc=dict(Geometry_HC=r['choices']['Geometry_HC'],IC0=int(old['choice'][0]));pred={}
            for name in ['QE','QS']:
                xx=np.concatenate([base,sf[:,:20 if name=='QE'else 46]],-1).astype(np.float32);q=models[name].inplace_predict(xx);raw={h:old['raw_'+h][0]for h in ['q','qp','qttc','qlk']};raw['q']=q;pp=project(raw);score=(5*pp[:,1]+5*pp[:,2]+2*pp[:,3]+2*a['features198'][:,197]*pp[:,0])/14;cc[name]=int(np.where(old['eligible'][0],score,-np.inf).argmax());pred.update({name+'_raw_q':q,name+'_projected':pp,name+'_score':score})
            ref=int(np.flatnonzero(a['valid'])[0]);assert ref==r.get('reference',ref)
            rr=dict(scene_id=sid,drive=r['drive'],group=r['group'],origin=r['origin'],source=r['source'],source_folder=str(src),reference=ref,choices=cc,candidates_to_evaluate=sorted(set([ref,*cc.values()])),source_receipt_sha=sha(src/'receipt.json'),source_input_sha=sha(src/'inputs.npz'),candidate_sha=__import__('hashlib').sha256(a['candidates'].tobytes()).hexdigest());rows.append(rr)
            np.savez_compressed(folder/'predictions.npz',signal_features=sf,feature_names=np.array(NAMES),valid=a['valid'],candidates=a['candidates'],eligible=old['eligible'][0],old_choice=old['choice'],**pred,**detail);write(folder/'signal_geometry.json',geometry);write(folder/'selection.json',rr)
        write(out/'choices.json',rows);write(out/'selection_freeze.json',dict(created_utc=now(),scope='old DEVELOPMENT140,6mappings,3drives; no new confirmation',main_freeze_sha=sha(BASE/'MAIN_FREEZE.json'),model_bindings=bindings,methods=METHODS,plan_sha=sha(OLD/'FULL_INPUT_LOCAL.json'),source_sha=sha(__file__),all_choices_before_new_outcomes=True,choices={r['scene_id']:r['choices']for r in rows}));print('B20_FULL_FREEZE',len(rows),sum(len(r['candidates_to_evaluate'])for r in rows),flush=True)
    from coordiworld.revision_v5.idm_teacher_geometry import install_corrected_builder
    install_corrected_builder()
    import nuplan.planning.simulation.observation.idm.idm_agent as idm
    from diagnose_idm_path import retry_dropped_endpoint
    from coordiworld.navsim_adapter import OfficialComponentScorer
    oldpath=idm.create_path_from_se2;retries=[];idm.create_path_from_se2=lambda poses:retry_dropped_endpoint(oldpath,poses,retries,True)
    owner=OfficialComponentScorer(ROOT/'data/maps',out/'metric_cache','reactive');cap=Capture(owner.scorer)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('B20native300sec')))
    for n,r in enumerate(rows):
        folder=out/'scenes'/r['scene_id']
        if(folder/'native_receipt.json').exists():continue
        signal.alarm(300);tt=time.monotonic();item=r['source'];assert sha(item['server_path'])==item['sha256'];inp=np.load(Path(r['source_folder'])/'inputs.npz')
        if r['origin']=='original':
            frames=b.load_log(item['server_path']);j=item['current_index'];scene=Scene.from_scene_dict_list(frames[j-3:j+9],None,4,8,SensorConfig.build_no_sensors())
        else:scene=Scene.load_from_disk(Path(item['server_path']),None,SensorConfig.build_no_sensors())
        cache=owner.processor.compute_metric_cache(NavSimScenario(scene,str(ROOT/'data/maps'),'nuplan-maps-v1.0'));checks=[];human=None
        for cid in r['candidates_to_evaluate']:
            poses=inp['candidates'][cid].astype(float);pilot.reset_rng(17);cap.calls=[];tab,states=pdm_score(copy.deepcopy(cache),Trajectory(poses,owner.prediction_sampling),owner.sampling,owner.simulator,owner.scorer,owner.traffic);calls=copy.deepcopy(cap.calls);assert len(calls)==(2 if r['origin']=='original'else 1)
            if len(calls)==2:
                if human is not None:assert human==calls[1]['raw'],'human reference changed between candidates'
                human=calls[1]['raw']
            ep=relative_to_absolute_poses(cache.ego_state.rear_axle,[StateSE2(*poses[-1])])[0];tab['valid']=True;tab['token']=r['scene_id'];tab['log_name']=cache.log_name;tab['frame_type']=cache.scene_type;tab['start_time']=cache.timepoint.time_s;tab['endpoint_x']=ep.x;tab['endpoint_y']=ep.y;tab['executed_endpoint_x']=states[-1,0];tab['executed_endpoint_y']=states[-1,1];tab['start_point_x']=cache.ego_state.rear_axle.x;tab['start_point_y']=cache.ego_state.rear_axle.y;tab['ego_simulated_states']=[states]
            oldfile=OLD/'full_primary/scenes'/r['scene_id']/f'candidate_{cid}_raw.pkl'
            if oldfile.exists():
                prev=pd.read_pickle(oldfile);err=float(np.abs(prev[METRICS[:8]].to_numpy(float)-tab[METRICS[:8]].to_numpy(float)).max());assert err<1e-7,('old score changed',r['scene_id'],cid,err);checks.append(dict(candidate=cid,old_sha=sha(oldfile),max_component_error=err))
            tab.to_pickle(folder/f'candidate_{cid}_filtered.pkl');write(folder/f'candidate_{cid}_physical.json',dict(candidate=cid,calls=calls,filtered={k:float(tab.iloc[0][k])for k in METRICS[:8]},endpoint_source='submitted trajectory.poses[-1] transformed from current rear axle',endpoint=[ep.x,ep.y],executed_endpoint=states[-1,:2].tolist(),trajectory_sha=__import__('hashlib').sha256(inp['candidates'][cid].tobytes()).hexdigest()))
        signal.alarm(0);del scene,cache;gc.collect();write(folder/'native_receipt.json',dict(seconds=time.monotonic()-tt,old_label_checks=checks,reference=r['reference'],human_fixed=True));write(out/'progress.json',dict(completed=n+1,total=len(rows),seconds=time.monotonic()-t0));print('B20_NATIVE',n+1,r['scene_id'],round(time.monotonic()-tt,2),flush=True)
    write(out/'path_end_retries.json',retries);mapping={(g['initial_token'],g['followup_token']):[tuple(p)for p in g['synthetic_initial_followup_pairs']]for g in plan['groups']};summary={}
    for m in METHODS:
        raw=pd.concat([pd.read_pickle(out/'scenes'/r['scene_id']/f'candidate_{r["choices"][m]}_filtered.pkl')for r in rows],ignore_index=True);raw.to_pickle(out/(m+'-native.pkl'));scored,sm=aggregate_complete(raw,mapping,owner.sampling);scored.to_pickle(out/(m+'-scored.pkl'));summary[m]=sm
        for g in plan['groups']:
            mm={(g['initial_token'],g['followup_token']):[tuple(p)for p in g['synthetic_initial_followup_pairs']]};tokens={s for pair,pp in mm.items()for p in[pair,*pp]for s in p};_,ss=aggregate_complete(raw[raw.token.isin(tokens)].copy(),mm,owner.sampling);write(out/('group'+str(g['registered_group_index'])+'-'+m+'.json'),ss)
    write(out/'result.json',dict(complete=True,scenes=140,mappings=6,drives=3,methods=summary,seconds=time.monotonic()-t0,scope='already developed, official submitted endpoint / own executed overlap EC',raw_physical_components_saved=True));print('B20_FULL_RESULT',{m:100*v['combined']['score']for m,v in summary.items()},flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:write(BASE/'development_full/failure.json',dict(created_utc=now(),traceback=traceback.format_exc()));raise
