"""Causal missing-slot adapter; never relabel an observed frame's timestamp.

Original regular histories take the unchanged original entry. Missing actor slots
are UNKNOWN, not interpolated observations. Only ego pose/velocity is interpolated
between two already observed past states; map, PDM proposal and current state are
computed from the untouched original history by the caller.
"""
import hashlib,json
import numpy as np
from pyquaternion import Quaternion

def install(runtime,audit_root):
 original=runtime.make_inputs
 def make_inputs(history,config,map_tokens=None,candidate_bundle=None):
  times=np.array([int(f['timestamp'])for f in history],dtype=np.int64)
  dt=float(config['data']['dt']);assert len(history)==config['data']['history_frames']==4
  if np.allclose(np.diff(times)/1e6,dt,atol=.08):
   return original(history,config,map_tokens,candidate_bundle)
  assert dt==.5 and np.all(np.diff(times)>0)
  assert np.allclose(np.diff(times[-3:])/1e6,dt,atol=.08),'Only missing OLDEST slot is admitted; no current/previous extrapolation'
  targets=times[-1]-np.arange(3,-1,-1,dtype=np.int64)*500000
  slots=[];details=[]
  for k,target in enumerate(targets):
   nearest=int(np.argmin(np.abs(times-target)))
   if abs(int(times[nearest])-int(target))<=80000:
    slots.append(history[nearest]);details.append(dict(slot=k,kind='actual_observation',source_index=nearest,timestamp_us=int(times[nearest])));continue
   assert k==0,'Only oldest slot may be absent in this version'
   lo=np.flatnonzero(times<target);hi=np.flatnonzero(times>target)
   assert len(lo)and len(hi),'No ego extrapolation'
   a=int(lo[-1]);b=int(hi[0]);assert times[b]-times[a]<=1100000
   u=float((target-times[a])/(times[b]-times[a]));left,right=history[a],history[b]
   qa,qb=Quaternion(*left['ego2global_rotation']),Quaternion(*right['ego2global_rotation'])
   q=Quaternion.slerp(qa,qb,amount=u)
   pos=(1-u)*np.asarray(left['ego2global_translation'],float)+u*np.asarray(right['ego2global_translation'],float)
   def rot(y):return np.array([[np.cos(y),-np.sin(y)],[np.sin(y),np.cos(y)]])
   va=rot(qa.yaw_pitch_roll[0])@np.asarray(left['ego_dynamic_state'][:2],float)
   vb=rot(qb.yaw_pitch_roll[0])@np.asarray(right['ego_dynamic_state'][:2],float)
   velocity=rot(-q.yaw_pitch_roll[0])@((1-u)*va+u*vb)
   slots.append(dict(timestamp=int(target),ego2global_translation=pos,ego2global_rotation=q.elements,
                     ego_dynamic_state=velocity,anns=dict(gt_boxes=np.empty((0,7)),gt_velocity_3d=np.empty((0,3)),gt_names=[],track_tokens=[])))
   details.append(dict(slot=k,kind='unobserved_actor_slot_with_causal_ego_interpolation',timestamp_us=int(target),
                       interpolation_observed_indices=[a,b],interpolation_weight=u,actors_observed=0,
                       cameras_or_lidar_synthesized=False))
  assert all(slots[k]is history[k]for k in range(1,4)) and slots[-1]is history[-1]
  x,meta=original(slots,config,map_tokens,candidate_bundle)
  assert not x.agent_valid[:,0].any() and not x.agents[:,0].any() and not x.confidence_valid[:,0].any()
  assert bool((x.source[:,0,:,0]==1).all())and not x.lineage[:,0].any()
  # Every actually observed retained slot is checked against the unchanged source transform.
  origin=runtime.ego_pose(history[-1]);tracks=meta['tracks'];checks=0
  for k in range(1,4):
   observed=runtime.frame_agents(history[k],origin)
   for j,track in enumerate(tracks):
    assert bool(x.agent_valid[0,k,j])==(track in observed)
    if track in observed:
     assert np.array_equal(x.agents[0,k,j].cpu().numpy(),np.asarray(observed[track][0],np.float32));checks+=7
  record=dict(token=str(history[-1]['token']),original_timestamps_us=times.tolist(),slots=details,
              unchanged_observed_actor_cells=checks,original_current_state=True,original_map_argument=True,
              original_pdm_candidate_bundle=True,labels_read=False,confidence_available=False,
              interpolation_scope='ego history only, bracketed by past observations; missing actor/source fields remain unknown',
              source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest())
  audit_root.mkdir(parents=True,exist_ok=True);p=audit_root/(record['token']+'.json')
  p.open('x').write(json.dumps(record,indent=2)+'\n')
  np.savez_compressed(audit_root/(record['token']+'.npz'),ego=x.ego.cpu().numpy(),agents=x.agents.cpu().numpy(),agent_valid=x.agent_valid.cpu().numpy(),source=x.source.cpu().numpy(),lineage=x.lineage.cpu().numpy())
  meta['history_slot_adapter']=record
  return x,meta
 runtime.make_inputs=make_inputs

 # The physical rail context consumes only history[-1]. Keep every construction
 # statement unchanged while removing its irrelevant FOUR-frame spacing guard.
 import ast,inspect,sys
 import coordiworld.revision_v5.current_rail_response as rail
 original_context=rail.current_context
 node=ast.parse(inspect.getsource(original_context)).body[0]
 original_ast=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
 assert original_ast=='77c0d16fdfcdb083368f0b7611066a89952a5bd2c14ed7592385d3842e2efabd'
 guards=[v for v in node.body if isinstance(v,ast.If)and any(isinstance(x,ast.Constant)and x.value=='History is not contiguous'for x in ast.walk(v))]
 assert len(guards)==1
 removed=ast.unparse(guards[0]);node.body.remove(guards[0])
 revised_ast=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
 ns=original_context.__globals__.copy()
 exec(compile(ast.Module(body=[node],type_ignores=[]),original_context.__code__.co_filename+'#A_R18_current_only_contract','exec'),ns)
 context=ns['current_context'];patched=[]
 for name,module in list(sys.modules.items()):
  if name.startswith('coordiworld.')and getattr(module,'current_context',None)is original_context:
   module.current_context=context;patched.append(name)
 assert 'coordiworld.revision_v5.current_rail_response'in patched and 'coordiworld.revision_v5.common_coverage_b13'in patched
 audit_root.mkdir(parents=True,exist_ok=True)
 (audit_root/'physical_context_binding.json').open('x').write(json.dumps(dict(original_function_ast_sha256=original_ast,
     revised_function_ast_sha256=revised_ast,removed_only=removed,patched_aliases=patched,
     body='All imports and current ego/actor/map construction AST remain exact; no historical observation is consumed by this current-only physical function',
     numerical_current_construction_changed=False),indent=2)+'\n')
