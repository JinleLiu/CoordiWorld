"""Measured batch1 current-history inference, with explicit scope/identity.
Loads only TRAIN examples, not protected datasets. I/O/model load separated.
"""
from revision_v5_twentythird_common import *
from revision_v5_twentythird_features import *
from revision_v5_twentythird_factory import continuation,ec_matrix
from revision_v5_twentythird_policy import PNC
from revision_v5_twentythird_null_probe import current_profile_candidate
from revision_v5_twentythird_spatial_probe import spatial_dynamic
from revision_v5_twentythird_progress_readout import select as hp_select
from revision_v5_twentysecond_pool_probe import brake
from revision_v5_twentieth_pool_probe import slot
from coordiworld.revision_v5.native_adapter import read_legacy_inputs
from coordiworld.revision_v5.compiled_solve_controller import predict_vector_controller_execution
import yaml,time,platform,torch
def infer(pipe,model,h,source_sha,previous=None,previous_timestamp=None,scene=None,meta=None):
 t0=time.perf_counter();base=pipe.inputs(h,source_sha,scene,meta);t1=time.perf_counter();ss,view=fix_s64(base['assets'],base['current'],base['S64'],base['S64_trace']);x=base['x'][0].copy();x[:,594:658]=ss;c32,_=contact_features(base['assets'],base['current'],base['areas']);a=dict(candidates=base['candidates'],valid=base['valid'],geo=np.array(base['Geometry_HC']),features198=base['features198'],x674=x,old_x674=base['x'][0],V24=vf(base['assets'])[0],R8=rf(base['areas'],base['valid']),C32=c32);asset=base['assets'];native=view['execution_native'];supply=False;continuation_added=False;generator_seconds=0.
 if (((c32[:,0]>0)|(c32[:,1]>0))|~a['valid']).all():
  ref=int(a['valid'].argmax());j,_=slot(a['candidates'],a['valid'],ref);dy=h[-1]['ego_dynamic_state'];ca,_=brake(float(dy[0]),float(dy[2]),a['candidates'][ref],6,6);a,asset,native,rr=add_candidate(pipe,h,source_sha,a,asset,base['current'],ca.astype(np.float32));a,asset=replace_from_appended(a,asset,j);native=recover_legacy(asset['execution_full_local'],base['current']['rear_axle_pose']);supply=True
 t2=time.perf_counter();ec=None;off=None
 if previous is not None:off=round((h[-1]['timestamp']-previous_timestamp)/1e5);ec=ec_matrix(native,previous,off)
 p=model.predict(a,ec)
 variant_steps=[]
 if ec is not None and not(ec[a['valid']]>0).any():
  original_native=native;baseline_choice=p['choice']
  for variant in ['history','current','spatial']:
   t=time.perf_counter()
   if variant=='history':ca,_=continuation(original_native[int(a['valid'].argmax()),0],previous,int(h[-1]['timestamp']),previous_timestamp,2.,'dynamic')
   elif variant=='current':ca,_=current_profile_candidate(original_native,baseline_choice,int(h[-1]['timestamp']),previous_timestamp)
   else:ca,_=spatial_dynamic(original_native[int(a['valid'].argmax()),0],previous,off)
   dup=any(np.linalg.norm(ca[:,:2]-old[:,:2],axis=1).max()<1e-3 and abs(np.arctan2(np.sin(ca[:,2]-old[:,2]),np.cos(ca[:,2]-old[:,2]))).max()<1e-4 for old in a['candidates'][a['valid']]);vr=dict(variant=variant,duplicate=dup,admitted=False)
   if not dup:
    _,ns,_=predict_vector_controller_execution(h,ca[None],np.ones(1,bool));ce=ec_matrix(ns[0][None],previous,off);vr['EC']=float(ce[0])
    if ce[0]>0:a,asset,native,rr=add_candidate(pipe,h,source_sha,a,asset,base['current'],ca);ec=ec_matrix(native,previous,off);continuation_added=True;vr.update(admitted=True,feature_seconds=rr['seconds'])
   vr['seconds']=time.perf_counter()-t;variant_steps.append(vr)
  p=model.predict(a,ec);p['choice']=hp_select(a|{'EC':ec}|{'PNC_'+k:v for k,v in p.items()})
 generator_seconds=sum(v['seconds']for v in variant_steps)
 torch.cuda.synchronize();t3=time.perf_counter();rec=dict(total_ms=1000*(t3-t0),base_world_map_execution_features_ms=1000*(t1-t0),corrected_features_and_brake_supply_ms=1000*(t2-t1),selection_and_history_supply_ms=1000*(t3-t2),history_generator_execution_ms=1000*generator_seconds,brake_supply=supply,continuation_added=continuation_added,choice=p['choice'],candidates=len(a['valid']),currenttimestamp=int(h[-1]['timestamp']),world_entities=int(asset['actor_valid'].sum()),variant_steps=variant_steps,selected_new=p['choice']>=12)
 return native[p['choice']],a,rec
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--memo',action='store_true');args=ap.parse_args();out=BASE/('timing_active_memo'if args.memo else'timing_active');out.mkdir(exist_ok=True);reg=json.load(open(BASE/'probe80/oof/result.json'));selected=[r for r in reg['rows']if r['scene_id']in ['study_scene_004898','study_scene_003020','study_scene_001383','study_scene_001699']];warmstart=time.perf_counter();pipe=Pipeline(yaml.safe_load(open(ROOT/'runtime_s1/configs/group_cache_v2.yaml')));models={k:PNC('oof',k)for k in range(3)}
 if args.memo:
  from revision_v5_twentythird_memo import install
  install(pipe)
 load_seconds=time.perf_counter()-warmstart;examples=[]
 for r in selected:
  so=r['source'];frames=b.load_log(ROOT/'data/openscene/trainval'/(so['log_name']+'.pkl'));j=so['index'];examples.append((r,frames[j-4:j],frames[j-3:j+1]))
 logs=[];identities=[]
 # Genuine fresh current-input proposal generation; no frozen-feature cache.
 for repeat in range(3):
  for r,ph,h in examples:
   model=models[r['fold']];prior,a,rr=infer(pipe,model,ph,r['source']['raw_log_sha256']);rr.update(scene_id=ph[-1]['token'],repeat=repeat,role='previous',mode='fresh_current_pipeline');logs.append(rr);native,a,rr=infer(pipe,model,h,r['source']['raw_log_sha256'],prior,int(ph[-1]['timestamp']));rr.update(scene_id=r['scene_id'],repeat=repeat,role='current',mode='fresh_current_pipeline');logs.append(rr)
   if repeat==0:
    old,_=read_legacy_inputs(r['source']['cache_path'],r['source']);oc=old.candidates[0].numpy();expected=np.load(BASE/'probe80/oof'/r['scene_id']/'inputs.npz')['previous_native'];np.testing.assert_allclose(prior,expected,atol=1e-6,rtol=0);identities.append(dict(scene_id=r['scene_id'],base_proposal_bytes_exact=bool(np.array_equal(a['candidates'][:12],oc)),base_valid_exact=bool(np.array_equal(a['valid'][:12],old.candidate_valid[0].numpy())),max_pose_difference=float(abs(a['candidates'][:12]-oc).max()),note='brake substitution can explain a changed base row; this timing run is not scored'))
   print('B23_TIMING',repeat,r['scene_id'],round(rr['total_ms'],2),flush=True)
 v=np.array([r['total_ms']for r in logs if r['repeat']>0 and r['role']=='current']);write(out/'result.json',dict(complete=True,created_utc=now(),source_sha=sha(__file__),records=logs,proposal_identity=identities,summary=dict(n=len(v),p50_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),max_ms=float(v.max()),mean_ms=float(v.mean())),model_pipeline_load_seconds=load_seconds,scope='batch1 four-current-history frames to own issued plan: PDM proposal+map+fixedworld+executor+S64/P16/C32/V24/R8+sixheads+triggered brake/historycandidate; I/O andmodel loading excluded;firstrepeatwarmupexcluded; summarycurrentframesonly; memo='+str(args.memo)+'',hardware=dict(GPU=torch.cuda.get_device_name(),CPU=platform.processor(),torch_threads=torch.get_num_threads()),limitation='fourTRAINactive-branch pairs, OOFsixheads samecapacity, exactpriorpredictionchecked; GPU/CPU structured-input planning chain only, no camera/V2X/network/native-scoring latency; no realroad realtime claim'))
if __name__=='__main__':main()
