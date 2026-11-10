"""Frozen PNC/M12/world bridge. All histories are policy-owned causal buffers."""
import json,time,hashlib,copy
from pathlib import Path
import numpy as np,yaml
ROOT=Path('<CW_WORKSPACE>');B=ROOT/'twentyeighth_20261031'
def digest(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def array_history(h):
 for f in h:
  f['cams']={k:{}for k in ['CAM_F0','CAM_L0','CAM_L1','CAM_L2','CAM_R0','CAM_R1','CAM_R2','CAM_B0']}
  an=f['anns'];an['gt_boxes']=np.asarray(an['gt_boxes'],float).reshape(-1,7);an['gt_velocity_3d']=np.asarray(an['gt_velocity_3d'],float).reshape(-1,3)
  for k in ['gt_names','track_tokens','instance_tokens']:an[k]=np.asarray(an[k])
  for k in ['ego2global_translation','ego2global_rotation','ego_dynamic_state','driving_command']:f[k]=np.asarray(f[k])
 return h
class BridgePolicy:
 def __init__(self,map_data,method,out,nominal_mode='planned'):
  self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.method=method
  from revision_v5_twentyeighth_map import install as install_map
  self.map=install_map(map_data)
  from revision_v5_twentyfourth_prepare import Pipeline,install,base_inputs,add_candidate,predict_vector_controller_execution
  from revision_v5_twentythird_policy import PNC
  self.pipe=install(Pipeline(yaml.safe_load(open(ROOT/'runtime_s1/configs/group_cache_v2.yaml'))));self.nominal_mode=nominal_mode
  if nominal_mode=='controlled_v02':
   from revision_v5_twentyeighth_nominal import install as install_nominal
   install_nominal(self.pipe)
  self.model=PNC();self.base_inputs=base_inputs;self.add_candidate=add_candidate;self.execution=predict_vector_controller_execution
  self.prior=None;self.prior_ts=None;self.n=0
  original=self.pipe.inputs
  def observed(*args,**kw):
   b=original(*args,**kw);self.last_base=b;return b
  self.pipe.inputs=observed
  (self.out/'model_identity.json').write_text(json.dumps(dict(world_sha=self.pipe.checkpoint_sha256,heads=self.model.bindings,method=method,execution='frozen NAVSIM Pacifica surrogate; real CARLA actuation uses shared PID tracker',history='own previously issued surrogate execution; current and past simulator observations only'),indent=2)+'\n')
 def plan(self,history):
  from revision_v5_twentythird_factory import continuation,ec_matrix
  from revision_v5_twentythird_spatial_probe import spatial_dynamic
  from revision_v5_twentythird_null_probe import current_profile_candidate
  from revision_v5_twentythird_progress_readout import select as pnc_select
  t=time.perf_counter();source=hashlib.sha256(json.dumps(history,sort_keys=True).encode()).hexdigest();h=array_history(copy.deepcopy(history))
  a,asset,native,current,receipt=self.base_inputs(self.pipe,h,source);base_seconds=time.perf_counter()-t
  ts=int(current['timestamp_us']);ec=np.ones(len(a['valid']));off=None;variants=[];trigger=False
  if self.prior is not None:
   off=round((ts-self.prior_ts)/1e5)
   if not(0<off<len(self.prior)-3):raise ValueError('Own plan history has no valid overlap')
   ec=ec_matrix(native,self.prior,off)
  pred=self.model.predict(a,None if off is None else ec);parent=int(pred['choice']);initial_valid=int(a['valid'].sum());ref=int(a['valid'].argmax())
  if off is not None and self.method=='Full':
   trigger=not bool((ec[a['valid']]>0).any())
   if trigger:
    for variant in ['history','current','spatial']:
     if variant=='history':ca,detail=continuation(native[ref,0],self.prior,ts,self.prior_ts,2.,'dynamic')
     elif variant=='current':ca,detail=current_profile_candidate(native,parent,ts,self.prior_ts)
     else:ca,detail=spatial_dynamic(native[ref,0],self.prior,off)
     duplicate=any(np.linalg.norm(ca[:,:2]-old[:,:2],axis=1).max()<1e-3 and abs(np.arctan2(np.sin(ca[:,2]-old[:,2]),np.cos(ca[:,2]-old[:,2]))).max()<1e-4 for old in a['candidates'][a['valid']])
     vr=dict(variant=variant,duplicate=duplicate,admitted=False,detail=detail)
     if not duplicate:
      _,fn,_=self.execution(h,ca[None],np.ones(1,bool));ce=float(ec_matrix(fn[0][None],self.prior,off)[0]);vr['EC']=ce
      if ce>0:
       a,asset,native,ar=self.add_candidate(self.pipe,h,source,a,asset,current,ca);np.testing.assert_allclose(native[-1],fn[0],atol=1e-7,rtol=0);ec=ec_matrix(native,self.prior,off);vr.update(admitted=True,candidate_index=len(a['valid'])-1,feature_receipt=ar)
     variants.append(vr)
   pred=self.model.predict(a,ec)
  ch=int(pred['choice'])
  if self.method=='Full'and off is not None:ch=pnc_select(a|{'EC':ec}|{'PNC_'+k:v for k,v in pred.items()})
  assert a['valid'][ch] and np.isfinite(a['x674'][a['valid']]).all()
  prior=self.prior;prior_ts=self.prior_ts;self.prior=np.array(native[ch],copy=True);self.prior_ts=ts
  row=dict(replan=self.n,method=self.method,timestamp_us=ts,source_sha=source,choice=ch,reference=ref,parent_choice=parent,Geometry_HC=int(a['geo']),initial_valid=initial_valid,valid=int(a['valid'].sum()),candidate_count=len(a['valid']),history_timestamp_us=prior_ts,history_offset=off,history_sha=None if prior is None else digest(prior),chosen_candidate_sha=digest(a['candidates'][ch]),chosen_execution_sha=digest(native[ch]),EC=float(ec[ch])if off is not None else None,trigger=trigger,variants=variants,base_seconds=base_seconds,seconds=time.perf_counter()-t,base_receipt=receipt,map_error=self.last_base.get('map_error'),pdm_error=self.last_base.get('PDM_error'),pdm_identity=self.last_base.get('PDM_identity'),source_keys=list(self.last_base))
  np.savez_compressed(self.out/f'plan_{self.n:05d}.npz',**a,native=native,EC=ec,**{'PNC_'+k:v for k,v in pred.items()},previous_native=np.zeros((0,11))if prior is None else prior,choice=ch)
  if self.n==0:
   if self.nominal_mode=='controlled_v02':
    nm=self.pipe.proposer.proposer;np.savez_compressed(self.out/'first_controlled_nominal.npz',raw_reference=nm.raw,executed_nominal=nm.executed)
   (self.out/'first_metadata.json').write_text(json.dumps(self.last_base['metadata'],default=lambda x:x.tolist()if hasattr(x,'tolist')else str(x),indent=2)+'\n')
   np.savez_compressed(self.out/'first_world_assets.npz',**{k:v for k,v in asset.items()if isinstance(v,np.ndarray)})
   (self.out/'first_current.json').write_text(json.dumps(current,indent=2)+'\n')
  with open(self.out/'plans.jsonl','a')as f:f.write(json.dumps(row,default=lambda x:x.tolist()if hasattr(x,'tolist')else str(x))+'\n')
  self.n+=1;return a['candidates'][ch],native[ch],row
if __name__=='__main__':
 import sys
 fi=json.load(open(sys.argv[1]));p=BridgePolicy(json.load(open(fi['map_path'])),'Full',B/'bridge/fixture_full');ca,ex,r=p.plan(fi['history']);print('ACTUAL_HISTORY_INTERFACE',r,flush=True)
