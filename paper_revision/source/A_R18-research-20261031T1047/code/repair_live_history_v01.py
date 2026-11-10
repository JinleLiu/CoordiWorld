"""Causal real-entry continuation of the two fixed post-selection repairs.
Reuse a saved bank only if its own-history H3 generator remains valid.
Re-enter the frozen current-input pipeline at affected H3 frames; propagate
the actual new issued plan, stop recomputation when the histories coincide.
"""
from coordiworld_publication import publication_identifier
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import argparse,ast,json,time,hashlib,gc,shutil,resource
import numpy as np,torch
import pnc_inference_input_fallback_v06 as pnc
import common_runtime as cr
from current_static_geometry_v01 import stopped_vehicle_guard
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
ap=argparse.ArgumentParser();ap.add_argument('--policy',choices=['static_guard','static_guard_absDAC'],required=True);args=ap.parse_args()
N=pnc.N;m=pnc.m;write=pnc.write;sha=pnc.sha
name=args.policy+'_live_history_v01';method='CoordiWorld-PNC_H3P_'+('staticGuard'if args.policy=='static_guard'else'staticGuard_absDAC')+'_live_v01'
plan=json.load(open(N/'queue_plans'/(name+'.json')));assert sha(__file__)==plan['tasks'][0]['source_sha256']
out=N/'analysis'/name;out.mkdir(exist_ok=False);started=time.time();vehicle=get_pacifica_parameters()
assert sha(N/'code/current_static_geometry_v01.py')==plan['geometry_source_sha256']
torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory)
# Only expose already calculated full states/assets at the original return.
src=N/'code/peer_reference/revision_v5_twentythird_timing_active.py';assert sha(src)==plan['infer_source_sha256']
node=next(x for x in ast.parse(src.read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='infer')
original_ast=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
returns=[x for x in ast.walk(node)if isinstance(x,ast.Return)];assert len(returns)==1
assert ast.unparse(returns[0].value)=="(native[p['choice']], a, rec)"
returns[0].value=ast.Tuple(elts=[ast.Name(id=k,ctx=ast.Load())for k in ['native','a','rec','asset']],ctx=ast.Load());node.name='infer_all_for_causal_reentry'
ast.fix_missing_locations(node);exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),pnc.ns)
pipe=None;head=None;world_calls=0;head_batches=0;cache_resets=0;live_calls=0;selection=[];allrows=[];pins={};logs={}
def ensure_model():
 global pipe,head
 if pipe is not None:return
 pipe=m.Pipeline();head=pnc.FrozenPNC();exp=json.load(open(N/'code/pnc_expected_bindings.json'))
 assert pipe.checkpoint_sha256==exp['world']and{k:v['sha256']for k,v in head.bindings.items()}==exp['heads']
 wp=pipe.world.predict;hp=head.predict
 def wp_count(*a,**kw):
  global world_calls,cache_resets
  if torch.cuda.memory_reserved()>2**30:
   torch.cuda.synchronize();pipe.world.graphs.clear();pipe.world.factual_graphs.clear();gc.collect();torch.cuda.empty_cache();cache_resets+=1
  world_calls+=1;return wp(*a,**kw)
 def hp_count(*a,**kw):
  global head_batches
  head_batches+=1;return hp(*a,**kw)
 pipe.world.predict=wp_count;head.predict=hp_count
 write(out/'model_binding.json',dict(world_sha256=pipe.checkpoint_sha256,heads=head.bindings,original_bindings=pnc.bindings,
    original_infer_AST_sha256=original_ast,modified_infer_AST_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),
    only_AST_changes='function name and return tuple; numerical body unchanged'))
def link(a,b):
 try:os.link(a,b)
 except OSError:shutil.copyfile(a,b)
for block in plan['blocks']:
 cohort=N/'data'/f'cohort_block{block:03}.json';assert sha(cohort)==plan['cohort_pins'][cohort.name]
 rows=json.load(open(cohort))['records'];allrows.extend(rows);previous=None;previous_ts=None;previous_pose=None;previous_token=None
 for row in rows:
  assert time.time()<plan['deadline_unix'];token=row['token'];d=N/'predictions/CoordiWorld-PNC_H3P'/token;old=json.load(open(d/'receipt.json'))
  dst=N/'predictions'/method/token;dst.mkdir(parents=True,exist_ok=False);live=False
  if old.get('input_contract_fallback'):
   assert token==publication_identifier('source_scene_006')and not row['scored']
   for f in ['trajectory.npy','own_predicted_execution.npy']:link(d/f,dst/f)
   choice=0;rec=dict(status='PASS_INHERITED_FALLBACK',token=token,choice=choice,live_reentry=False,scored=False)
  else:
   if old['own_previous_timestamp']is None:previous=None;previous_ts=None;previous_pose=None;previous_token=None
   else:assert previous_ts==old['own_previous_timestamp']
   same_previous=previous is None or np.array_equal(previous_pose,np.load(N/'predictions/CoordiWorld-PNC_H3P'/previous_token/'trajectory.npy'))
   a=dict(np.load(d/'candidate_features.npz',allow_pickle=False));h=dict(np.load(d/'head_outputs.npz',allow_pickle=False))
   gr=json.load(open(N/'predictions/CoordiWorld-PNC_H3P_staticGuard_v01'/token/'receipt.json'));gp=Path(gr['source_geometry']);geo=dict(np.load(gp,allow_pickle=False));native=geo['execution_native'];unsafe=geo['unsafe']
   ec=None if previous is None else pnc.ns['ec_matrix'](native,previous,round((old['currenttimestamp']-previous_ts)/1e5))
   trigger_new=False if ec is None else not bool((ec[:12][a['valid'][:12]]>0).any())
   live=not same_previous and(bool(old['variant_steps'])or trigger_new)
   bankrec=old
   if live:
    ensure_model();path=row['raw_log']
    if path not in logs:assert sha(path)==row['raw_log_sha256'];logs={path:m.load_log(path)}
    history=logs[path][row['index']-3:row['index']+1];assert history[-1]['token']==token
    torch.cuda.synchronize();t0=time.perf_counter()
    native,a,bankrec,assets=pnc.ns['infer_all_for_causal_reentry'](pipe,head,history,row['raw_log_sha256'],previous,previous_ts)
    h={k:np.asarray(v).copy()for k,v in head.last_prediction.items()};live_calls+=1
    origin=np.asarray(m.ego_pose(history[-1]));cur={k:v for k,v in m.frame_agents(history[-1],origin).items()if np.linalg.norm(v[0][:2])<=100}
    states=np.asarray([v[0]for v in cur.values()],float).reshape(-1,7);types=np.asarray([v[1]for v in cur.values()],int)
    c,s=np.cos(origin[2]),np.sin(origin[2]);local=native[...,:3].copy();local[...,:2]=(native[...,:2]-origin[:2])@np.array([[c,-s],[s,c]]);local[...,2]=np.arctan2(np.sin(native[...,2]-origin[2]),np.cos(native[...,2]-origin[2]))
    geo=stopped_vehicle_guard(local,np.linalg.norm(native[...,3:5],axis=-1),states,types,np.ones(len(states),bool),a['valid'],[vehicle.length,vehicle.width],vehicle.rear_axle_to_center);unsafe=geo['unsafe']
    ec=None if previous is None else pnc.ns['ec_matrix'](native,previous,round((old['currenttimestamp']-previous_ts)/1e5))
    live_dir=out/'live_cases'/token;live_dir.mkdir(parents=True)
    np.savez_compressed(live_dir/'candidate_features.npz',**a);np.savez_compressed(live_dir/'head_outputs.npz',**h)
    np.savez_compressed(live_dir/'world_assets.npz',**{k:v for k,v in assets.items()if isinstance(v,np.ndarray)})
    np.savez_compressed(live_dir/'geometry.npz',execution_native=native,current=states,types=types,tokens=np.asarray(list(cur)),valid=a['valid'],**geo)
    torch.cuda.synchronize();write(live_dir/'receipt.json',dict(status='PASS',token=token,previous=previous_token,bank_receipt=bankrec,seconds=time.perf_counter()-t0,source_current_log_sha256=row['raw_log_sha256']))
   eligible=h['eligible'].copy();score=h['scores'].copy()if ec is None else(14*h['scores']+2*h['projected'][:,0]*ec)/16
   extra=np.arange(len(a['valid']))>=12
   if bankrec['variant_steps']:
    assert ec is not None
    eligible|=extra&a['valid']&(ec>0)&(h['raw_physical_nc']<=.5)&(h['raw_dac_event']<=.5)
    p0=extra&(a['x674'][:,668]>0);score[p0]+=5/16*(h['projected'][p0,0]*a['x674'][p0,662]-h['projected'][p0,1])
   if args.policy=='static_guard_absDAC':
    add=a['valid']&~unsafe&(h['raw_physical_nc']<=.5)&(h['raw_dac_event']<=.5)
    add&=~extra if ec is None else(~extra|(ec>0));eligible|=add
   choice=int(np.where(eligible,score,-np.inf).argmax());before=choice;clear=eligible&~unsafe
   if unsafe[choice]and clear.any():choice=int(np.where(clear,score,-np.inf).argmax())
   poses=a['candidates'][choice];own=native[choice]
   canonical='current_bank'
   if np.array_equal(poses,np.load(d/'trajectory.npy')):
    assert np.max(np.abs(own-np.load(d/'own_predicted_execution.npy')))<=1e-7
    own=np.load(d/'own_predicted_execution.npy');canonical='exact_original_trajectory_original_own_state'
   np.save(dst/'trajectory.npy',poses);np.save(dst/'own_predicted_execution.npy',own)
   np.savez_compressed(dst/'selector.npz',eligible=eligible,scores=score,unsafe=unsafe,own_EC=np.array([])if ec is None else ec)
   rec=dict(status='PASS',token=token,choice=choice,old_choice=int(old['choice']),scored=row['scored'],previous=previous_token,own_previous_timestamp=previous_ts,
    same_original_previous=same_previous,original_H3_trigger=bool(old['variant_steps']),recomputed_H3_trigger=bool(trigger_new),
    live_reentry=live,bank_candidates=len(a['valid']),guard_changed=choice!=before,canonical_own_state=canonical)
   for p in [d/'receipt.json',d/'candidate_features.npz',d/'head_outputs.npz',gp]:pins[str(p.relative_to(N))]=sha(p)
  rec.update(method=method,currenttimestamp=old['currenttimestamp'],trajectory_sha256=sha(dst/'trajectory.npy'),
    changed_from_original=not np.array_equal(np.load(dst/'trajectory.npy'),np.load(d/'trajectory.npy')))
  write(dst/'receipt.json',rec);selection.append(rec|{'drive':row['drive']})
  previous=np.load(dst/'own_predicted_execution.npy');previous_pose=np.load(dst/'trajectory.npy');previous_ts=old['currenttimestamp'];previous_token=token
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<6*2**30
  if len(selection)%100==0:write(out/'live.json',dict(phase='real_own_history',inputs=len(selection),total=2919,real_pipeline_calls=live_calls,world_entry_calls=world_calls,seconds=time.time()-started))
assert len(selection)==2919
write(out/'selection_frozen.json',dict(status='PASS',unix=time.time(),rows=selection,before_new_outcomes=True))
new=0;reuse=0;score_rows=[]
for row in allrows:
 assert time.time()<plan['deadline_unix'];token=row['token'];pred=N/'predictions'/method/token;poses=np.load(pred/'trajectory.npy');dest=N/'results'/method/token;dest.mkdir(parents=True,exist_ok=False);np.save(dest/'trajectory.npy',poses);parent=None
 for oldmethod in ['CoordiWorld-PNC_H3P_staticGuard_absDAC_v01','CoordiWorld-PNC_H3P_staticGuard_v01','CoordiWorld-PNC_H3P','GTRS-Dense-PNC-bank_staticGuard_v01','GTRS-Dense-PNC-bank','PDM-Closed']:
  p=N/'results'/oldmethod/token
  if(p/'score_raw.pkl').exists()and np.array_equal(np.load(p/'trajectory.npy'),poses):parent=p;break
 if parent is not None:
  for f in ['score_raw.pkl','execution.npy']:link(parent/f,dest/f)
  reuse+=1
 else:
  df,states=cr.score_one(row,poses,cr.load_cache(row));df.to_pickle(dest/'score_raw.pkl');np.save(dest/'execution.npy',states);new+=1
 err=float(np.max(np.abs(np.load(dest/'execution.npy')-np.load(pred/'own_predicted_execution.npy'))));assert err<=1e-7,(token,err)
 rec=dict(token=token,reused=str(parent)if parent else None,execution_max_error=err,trajectory_sha256=sha(dest/'trajectory.npy'));write(dest/'receipt.json',rec);score_rows.append(rec)
ad=out/method;ad.mkdir();summary=cr.aggregate(method,allrows,ad)
write(out/'score_reuse.json',score_rows);write(out/'input_pins.json',pins)
write(out/'result.json',dict(status='PASS',summary=summary,real_pipeline_calls=live_calls,world_entry_calls=world_calls,head_batches=head_batches,
 graph_cache_resets=cache_resets,changed_inputs=sum(x['changed_from_original']for x in selection),new_native_scores=new,reused_native_scores=reuse,
 training_updates=0,seconds=time.time()-started,RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 CUDA_allocated_peak=torch.cuda.max_memory_allocated(),CUDA_reserved_peak=torch.cuda.max_memory_reserved(),
 scope='Real current-input H3 regeneration where own issued-history differs; unchanged-input banks reused. Each policy actual adjacent executions define EC. Same frozen world/heads/thresholds, no development fitting.'))
print('LIVE_REPAIR_PASS',json.dumps(summary),flush=True)
