"""Frozen-policy replication on additional old development drives, NOT CONF.

Select cases without outcomes; save all choices before native scoring. Native
labels cover only the union of frozen selections plus reference. Candidate
pool/validity never changes. No oracle ranking or full EPDMS is inferred.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ['TORCHINDUCTOR_COMPILE_THREADS']='2'
import copy,datetime as dt,hashlib,json,multiprocessing,signal,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch,joblib
from coordiworld.model import ModelConfig
from coordiworld.data.openscene import load_log,ego_pose
from coordiworld.data.maps import CurrentMapProvider
from coordiworld.revision_v5.body_heading import BodyHeadingKinematicResidual,body_heading_states
from coordiworld.revision_v5.kinematic_residual import kinematic_prior
from coordiworld.revision_v5.absolute_planning_world import SweptOccupancyHybridWorld
from coordiworld.revision_v5.compiled_solve_controller import predict_vector_controller_execution
from coordiworld.revision_v5.native_adapter import read_legacy_inputs
from coordiworld.revision_v5.world_score_features import candidate_features
from revision_v5_eleventh_relation_features import relation_features
from revision_v5_eleventh_map_augment import static_features
from revision_v5_eleventh_factor_choice import pred
from current_progress_penalty import select_with_progress_penalty
from revision_v5_eleventh_choice_common import choose,probabilities,reporting_calibration,macro,bootstrap,risk_summary
from revision_v5_train_factual_metric_aligned import factual_inputs
from revision_v5_collect_paired_teachers_v2 import execution_rules
from current_map_export import export_current_map
from deployment_features import build_trace
import revision_v5_audit_current_rail as audit
ROOT=Path(__import__('coordiworld_paths').workspace_path(''));RUN=ROOT/'eleventh_20261026/runs';OUT=RUN/'new_scene_replication_progress_v3'
DEADLINE=dt.datetime.fromisoformat('2026-10-26T08:52:00+00:00').timestamp()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def initialize_scoring():
 global SCORER
 torch.set_num_threads(1)
 from threadpoolctl import threadpool_limits
 threadpool_limits(limits=1)
 from coordiworld.revision_v5.idm_teacher_geometry import install_corrected_builder
 install_corrected_builder()
 import nuplan.planning.simulation.observation.idm.idm_agent as idm
 from diagnose_idm_path import retry_dropped_endpoint
 old=idm.create_path_from_se2;events=[];idm.create_path_from_se2=lambda poses:retry_dropped_endpoint(old,poses,events,True)
 from coordiworld.navsim_adapter import OfficialComponentScorer
 SCORER=OfficialComponentScorer(ROOT/'data/maps',OUT/('metric_cache_'+str(os.getpid())),'reactive')
def score_row(record):
 sid=record['scene_id'];folder=OUT/sid;row=record['input'];started=time.monotonic()
 def alarm(*_):raise TimeoutError('Fixed per-scene native replication limit')
 signal.signal(signal.SIGALRM,alarm);signal.alarm(max(1,min(240,int(DEADLINE-time.time()))))
 try:
  import revision_v5_collect_native_pilot as pilot
  raw=ROOT/'data/openscene/trainval'/(row['log_name']+'.pkl');assert sha(raw)==row['raw_log_sha256'];frames=load_log(raw);assert str(frames[row['index']]['token'])==sid
  scene,meta=read_legacy_inputs(row['cache_path'],row);cache=SCORER.build_metric_cache(frames,row['index']);scores={}
  for j in record['label_candidates']:
   pilot.reset_rng(17);scores[str(j)]=SCORER.score(copy.deepcopy(cache),scene.candidates[0,j].numpy())
  pilot.reset_rng(17);repeat=SCORER.score(copy.deepcopy(cache),scene.candidates[0,record['reference']].numpy());assert repeat==scores[str(record['reference'])],'Reference repeat differs'
  result=dict(complete=True,scene_id=sid,scores=scores,reference_repeat_exact=True,seconds=time.monotonic()-started)
 except Exception as error:result=dict(complete=False,scene_id=sid,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started)
 finally:signal.alarm(0)
 audit.save(folder/'native_scores.json',result);return result
def main():
 OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();base=dict(np.load(RUN/'map_augmented_features_v1b/dataset.npz'));oldids=set(base['scene_ids']) | {r['scene_id'] for name in ['new_scene_replication_v1','new_scene_replication_plain_v2'] for r in json.loads((RUN/name/'pre_registration.json').read_text())['requested']};inventory=json.loads((ROOT/'runs/paired_geometry_v2_train128_dev64/input_screening.json').read_text())['rows'];oldrows=[r for r in inventory if r['scene_id'] in oldids]
 selected=[]
 for role in ['probe','development_validation']:
  drives=sorted(set(base['drives'][base['role_'+role]]),key=lambda s:h('b11-new-drive/'+s));groups={}
  for drive in drives:
   available=[r for r in inventory if r['drive']==drive and r['scene_id'] not in oldids and not any(r['log_name']==q['log_name'] and max(r['window_start'],q['window_start'])<=min(r['window_end'],q['window_end']) for q in oldrows)]
   groups[drive]=sorted(available,key=lambda r:h('b11-replication-progress-v3/'+r['scene_id']))
  taken=[]
  for position in range(100):
   for drive in drives:
    if position<len(groups[drive]) and len(taken)<32:taken.append({**groups[drive][position],'replication_role':role})
   if len(taken)==32:break
  assert len(taken)==32;selected+=taken
 definitions={'physical_no_action_progress':('physical_expanded_plain_v1','physical_no_action',False,'penalty'),'physical_no_action_raw':('physical_expanded_plain_v1','physical_no_action',False,None),'CTRA_progress':('factor_expanded_plain_v1','CTRA',False,'penalty'),'CTRA_plain_raw':('factor_expanded_plain_v1','CTRA',False,None)}
 policy_path=RUN/'progress_penalty_v1/result.json';policy=json.loads(policy_path.read_text());assert policy['has_eligible_policy'];penalty=policy['selected_drop'];assert penalty==.5
 models={};calibrations={};sources={str(Path(__file__)):sha(__file__),str(policy_path):sha(policy_path),str(ROOT/'eleventh_20261026/portable/current_progress_penalty.py'):sha(ROOT/'eleventh_20261026/portable/current_progress_penalty.py')}
 ci=base['role_head_calibration'];y=base['events'];cv=base['valid']
 for name,(folder,mode,with_map,margin) in definitions.items():
  path=RUN/folder/(mode+'_models.joblib');sources[str(path)]=sha(path);models[name]=joblib.load(path)
  pp=np.load(RUN/folder/(mode+'_predictions.npz'));g=pp['relative_tree'];ll=pp['logits'];choice=select_with_progress_penalty(g,cv,base['shared_geometry'],penalty)[0] if margin=='penalty' else (choose(g,ll,cv,margin) if margin is not None else np.where(cv,g,-np.inf).argmax(1));calibrations[name]=reporting_calibration(ll[ci],choice[ci],y[ci],base['drives'][ci])
 cp=ROOT/'runs/dynamics_residual_batch_v1/kinematic_residual_s17/unqualified_model.pt';sources[str(cp)]=sha(cp)
 audit.save(OUT/'pre_registration.json',dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),requested=selected,methods=definitions,primary='physical_no_action_progress; current-progress penalty selected TRAIN head-OOF only; rawphysical, matched CTRA+penalty, rawCTRA controls',current_progress_penalty_weight=penalty,source_sha256=sources,calibrations=calibrations,selection='32 scene identities per existing probe/DEV stratum, drive-balanced round robin and SHA order; no outcome/geometry selection; nonoverlap with old score windows AND previous128scene replications; fresh SHA order',independent_confirmation=False,reason='Same developed drives; CONF untouched. First two new cohorts exposed unnecessary slowdowns. Hard cap failed TRAIN-OOF; soft one-sided penalty weight.5 selected solely from TRAIN-OOF and frozen before this third new cohort.',predictions_before_native_scores=True,labels='union of frozen selected candidates and unchanged first-valid reference; full original candidate pool remains available',native_scope='same rear-axle-corrected reactive v2; filtered single-stage components only',deadline_utc='2026-10-26T08:52:00+00:00',workers=4,max_scene_seconds=240,full_EPDMS=False))
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch._dynamo.config.recompile_limit=128;ck=torch.load(cp,map_location='cpu',weights_only=True);model=BodyHeadingKinematicResidual(ModelConfig(**ck['model_config'])).cuda().eval().requires_grad_(False);model.load_state_dict(ck['state_dict']);audit.initialize(str(ROOT),str(OUT),[]);api=SweptOccupancyHybridWorld(ROOT/'data/maps',model);provider=CurrentMapProvider(ROOT/'data/maps');records=[];failures=[];frames=None;last=None
 with torch.inference_mode():
  for row in sorted(selected,key=lambda r:(r['log_name'],r['index'])):
   sid=row['scene_id'];folder=OUT/sid;folder.mkdir()
   try:
    assert time.time()<DEADLINE-300,'Prediction budget reserve';scene,meta=read_legacy_inputs(row['cache_path'],row);assert sha(row['cache_path'])==row['cache_sha256']
    if last!=row['log_name']:
     raw=ROOT/'data/openscene/trainval'/(row['log_name']+'.pkl');assert sha(raw)==row['raw_log_sha256'];frames=load_log(raw);last=row['log_name']
    history=frames[row['index']-3:row['index']+1];actual=api.predict(history,scene,meta['tracks'],factual_inputs(scene));world=actual['states'];ref=actual['reference'];execution,full,receipt=predict_vector_controller_execution(history,scene.candidates[0].numpy(),scene.candidate_valid[0].numpy());margins,rv=execution_rules(scene,meta,history,execution,provider);valid=scene.candidate_valid[0].numpy();ids=np.flatnonzero(valid)
    origin=np.array(receipt['coordinate_origin']);co,si=np.cos(origin[2]),np.sin(origin[2]);poses=np.stack([full[int(j)][:,:3] for j in ids]);poses[...,:2]=(poses[...,:2]-origin[:2])@np.array([[co,-si],[si,co]]);poses[...,2]-=origin[2];now=history[-1];dynamic=np.asarray(now['ego_dynamic_state'],float);current=dict(scene_id=sid,timestamp_us=int(now['timestamp']),latest_source_timestamp_us=int(now['timestamp']),map_name=now['map_location'],rear_axle_pose=ego_pose(now).tolist(),velocity=dynamic[:2].tolist(),acceleration=dynamic[2:].tolist(),route_roadblock_ids=list(now['roadblock_ids']),traffic_lights=[list(x) for x in now['traffic_lights']],source_container_sha256=row['raw_log_sha256']);geometry=np.zeros((12,18),np.float32);map_error=None
    try:
     bundle,coords,native=export_current_map(current,ROOT/'data/maps',poses);e,t=build_trace(bundle,poses);geometry[ids]=static_features(e,t)
    except Exception as error:map_error=repr(error)
    scene=scene.to('cuda');executed=torch.from_numpy(execution).cuda();mm=torch.from_numpy(margins).cuda();vv=torch.from_numpy(rv).cuda();ctra=body_heading_states(factual_inputs(scene),kinematic_prior(factual_inputs(scene))[:,None])[0,0];worlds={'CTRA':ctra[None].expand_as(world),'physical_action':world,'physical_no_action':world[ref:ref+1].expand_as(world)};features={};selections={};risks={};outputs={}
    for mode,w in worlds.items():
     plain=candidate_features(scene,executed,mm,vv,w).cpu().numpy();relation=relation_features(scene,executed,w).cpu().numpy();features[mode]=(plain,np.concatenate((plain,relation,geometry),-1))
    for name,(_,mode,with_map,margin) in definitions.items():
     x=features[mode][int(with_map)][None];gg,ll=pred(models[name],x,valid[None]);gain=gg['relative_tree'];choice=int(select_with_progress_penalty(gain,valid[None],geometry[None],penalty)[0][0]) if margin=='penalty' else (int(choose(gain,ll,valid[None],margin)[0]) if margin is not None else int(np.where(valid,gain[0],-np.inf).argmax()));selections[name]=choice;cl=ll[0,choice,8:].copy()
     for j,c in enumerate(calibrations[name]):cl[j]/=c['temperature']
     risks[name]=probabilities(cl).tolist();outputs[name+'_gain']=gain;outputs[name+'_logits']=ll
    record=dict(scene_id=sid,input=row,role=row['replication_role'],drive=row['drive'],reference=ref,selections=selections,risks=risks,label_candidates=sorted(set([ref]+list(selections.values()))),valid_candidates=ids.tolist(),map_error=map_error,map_known=bool(geometry[ids,0].all()))
    np.savez_compressed(folder/'predictions.npz',**outputs,geometry=geometry);audit.save(folder/'frozen_choices.json',record);records.append(record);print(json.dumps(dict(predicted=len(records),scene_id=sid,seconds=time.monotonic()-start)),flush=True)
   except Exception as error:failures.append(dict(scene_id=sid,error=repr(error),traceback=traceback.format_exc()));audit.save(folder/'prediction_failure.json',failures[-1])
 audit.save(OUT/'prediction_manifest.json',dict(records=records,failures=failures,requested=64,all_choices_frozen_before_native_labels=True,source_hashes_unchanged=all(sha(p)==digest for p,digest in sources.items())))
 del api,model;torch.cuda.empty_cache()
 scored=[]
 with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_scoring) as pool:
  futures=[pool.submit(score_row,r) for r in records]
  for future in as_completed(futures):
   try:scored.append(future.result())
   except Exception as error:scored.append(dict(complete=False,error=repr(error)))
   audit.save(OUT/'collection_progress.json',dict(completed=sum(r['complete'] for r in scored),returned=len(scored),attempted=len(records)));print(json.dumps(dict(native_returned=len(scored),completed=sum(r['complete'] for r in scored),seconds=time.monotonic()-start)),flush=True)
 good={r['scene_id']:r for r in scored if r['complete']};analyses={}
 for role in ['probe','development_validation']:
  rr=[r for r in records if r['role']==role and r['scene_id'] in good];ds=np.array([r['drive'] for r in rr]);reference=[good[r['scene_id']]['scores'][str(r['reference'])] for r in rr];ru=np.array([s['single_stage_composite_excluding_two_frame_comfort'] for s in reference]);rp=np.array([s['ego_progress'] for s in reference]);re=np.array([[s['no_at_fault_collisions']<1,s['drivable_area_compliance']<1] for s in reference],float);methods={}
  for name in definitions:
   yy=[good[r['scene_id']]['scores'][str(r['selections'][name])] for r in rr];u=np.array([s['single_stage_composite_excluding_two_frame_comfort'] for s in yy]);progress=np.array([s['ego_progress'] for s in yy]);events=np.array([[s['no_at_fault_collisions']<1,s['drivable_area_compliance']<1] for s in yy],float);risk=np.array([r['risks'][name] for r in rr]);switch=np.array([r['selections'][name]!=r['reference'] for r in rr]);gain=u-ru
   methods[name]=dict(score_points=100*macro(u,ds),reference_score_points=100*macro(ru,ds),gain_points=100*macro(gain,ds),exploratory_drive_CI_points=bootstrap(100*gain,ds),NC=int(events[:,0].sum()),DAC=int(events[:,1].sum()),reference_NC=int(re[:,0].sum()),reference_DAC=int(re[:,1].sum()),NC_new=int(((events-re)[:,0]>0).sum()),NC_rescued=int(((events-re)[:,0]<0).sum()),DAC_new=int(((events-re)[:,1]>0).sum()),DAC_rescued=int(((events-re)[:,1]<0).sum()),progress_delta=macro(progress-rp,ds),switches=int(switch.sum()),selected_risk_metrics=risk_summary(risk,events,ds,[c['constant'] for c in calibrations[name]]),scene_ids=[r['scene_id'] for r in rr],selected_utility=u.tolist(),selected_progress=progress.tolist(),selected_events=events.tolist(),selected_risk=risk.tolist(),reference_utility=ru.tolist(),reference_progress=rp.tolist(),reference_events=re.tolist())
  analyses[role]=dict(requested=32,completed=len(rr),drives=len(set(ds)),methods=methods)
 result=dict(complete=len(good)==64 and not failures,requested=64,predictions=len(records),prediction_failures=failures,native_failures=[r for r in scored if not r['complete']],analyses=analyses,seconds=time.monotonic()-start,independent_confirmation=False,full_EPDMS=False,no_training_or_policy_updates_after_freeze=True,source_hashes_unchanged=all(sha(p)==digest for p,digest in sources.items()))
 audit.save(OUT/'result.json',result);print(json.dumps(dict(complete=result['complete'],seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
