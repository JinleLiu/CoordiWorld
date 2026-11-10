"""The one permitted new policy run, S1, with genuine own-plan propagation.

Historical C1's causal bank-reentry condition and exact shared proposer are kept.
Only the final selector is replaced. No labels enter inference; evaluate after
all2919 selections are sealed. No clocks/deadlines inherited from old rounds.
"""
from coordiworld_publication import publication_identifier
import ast,copy,fcntl,gc,os,pickle,resource,signal,time,traceback
import experiment_common as e
import numpy as np

def lock(name):
 p=e.RUN/'locks'/name;p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f

def infer_function():
 source=e.R18/'code/peer_reference/revision_v5_twentythird_timing_active.py'
 tree=ast.parse(source.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='infer');orig=copy.deepcopy(node)
 assert ast.unparse(node.body[-1].value)=="(native[p['choice']], a, rec)"
 node.body[-1].value=ast.parse("(native,a,rec,asset,base['current'],j if supply else None)",mode='eval').body
 assert ast.dump(ast.Module(body=node.body[:-1],type_ignores=[]))==ast.dump(ast.Module(body=orig.body[:-1],type_ignores=[]))
 ns=dict(e.pnc.ns);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source)+'#return_diagnostics_only','exec'),ns)
 return ns['infer']

def generate():
 e.verify_implementation()
 held=lock('OWN_INFERENCE.lock');cfg=e.read(e.RUN/'configs/SIMPLE_FINAL_FREEZE_v01.json');rule=cfg['rule'];freeze=e.sha(e.RUN/'configs/SIMPLE_FINAL_FREEZE_v01.json')
 assert cfg['source_sha256']==e.sha(e.RUN/'code/simple_rules.py')
 assert e.read(e.RUN/'configs/CORE_CONTRIBUTION_FREEZE_v01.json')['path']=='B'
 assert e.read(e.RUN/'evidence/CONDITIONAL_COMPLETE.json')['status']=='PASS','Only one new full queue at a time'
 model=None;pipe=None;infer=None;live_calls=0;world_calls=0;logs={};records=[];started=time.monotonic()
 previous=previous_pose=previous_ts=previous_token=None;last_block=None
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1800s per-input resource watchdog, preserve outputs')))
 def ensure_model():
  nonlocal model,pipe,infer,world_calls
  if pipe is not None:return
  import torch
  torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory)
  audit=e.RUN/'evidence/runtime'/f'own_{os.getpid()}'
  from pnc_missing_history_slots_v02 import install
  install(e.pnc.m,audit)
  from current_controller_contract import install as cc
  cc(e.pnc.ns,audit)
  pipe=e.pnc.m.Pipeline();model=e.pnc.FrozenPNC()
  expected=e.read(e.R18/'code/pnc_expected_bindings.json')
  assert pipe.checkpoint_sha256==expected['world'] and {k:v['sha256'] for k,v in model.bindings.items()}==expected['heads']
  from bounded_graph_cache import install as cache_install
  cache=cache_install(pipe,limit=2)
  old_predict=pipe.world.predict
  def counted(*a,**kw):
   nonlocal world_calls
   world_calls+=1;return old_predict(*a,**kw)
  pipe.world.predict=counted;infer=infer_function()
  e.write(audit/'MODEL_BINDINGS.json',dict(at=e.now(),world_sha256=pipe.checkpoint_sha256,heads=model.bindings,
     proposer=e.pnc.bindings,cache=cache,GPU_limit_bytes=4*2**30,new_fits=0,labels_in_inference=False))
 rows=e.cohort()
 for i,row in enumerate(rows):
  t0=time.monotonic();token=row['token'];original=e.R18/'predictions'/e.C0/token;old=e.read(original/'receipt.json')
  if row['block']!=last_block:previous=previous_pose=previous_ts=previous_token=None;last_block=row['block']
  fallback=bool(old.get('input_contract_fallback'))
  if not fallback:
   if old['own_previous_timestamp'] is None:previous=previous_pose=previous_ts=previous_token=None
   else:assert previous_ts==old['own_previous_timestamp'],(token,'causal timestamp mismatch')
  history=dict(previous_token=previous_token,previous_timestamp=previous_ts,previous_execution_sha256=None if previous is None else e.hashlib.sha256(np.ascontiguousarray(previous).tobytes()).hexdigest(),block=row['block'])
  dest=e.RUN/'predictions'/e.S1/token;receipt=dest/'receipt.json'
  if receipt.exists():
   rec=e.read(receipt);assert rec['status']=='PASS' and rec['Simple_freeze_sha256']==freeze and rec['history']==history
   assert e.sha(dest/'trajectory.npy')==rec['trajectory_sha256'] and e.sha(dest/'own_predicted_execution.npy')==rec['execution_sha256']
   previous=np.load(dest/'own_predicted_execution.npy',allow_pickle=False);previous_pose=np.load(dest/'trajectory.npy',allow_pickle=False);previous_ts=rec['currenttimestamp'];previous_token=token
   records.append(dict(token=token,scored=bool(row['scored']),receipt_sha256=e.sha(receipt)));live_calls+=bool(rec['live_reentry']);continue
  if dest.exists():
   preserved=e.RUN/'failures/unsealed_inputs'/token/f'{os.getpid()}_{time.time_ns()}';preserved.parent.mkdir(parents=True,exist_ok=True);dest.rename(preserved)
  dest.mkdir(parents=True);signal.alarm(1800);live=False
  e.write(e.RUN/'queue/OWN_INFERENCE.json',dict(at=e.now(),status='RUNNING',pid=os.getpid(),done=i,total=len(rows),current=token,live_reentries=live_calls,world_calls=world_calls,seconds=time.monotonic()-started))
  if fallback:
   assert token==publication_identifier('source_scene_006') and not row['scored']
   poses=np.load(original/'trajectory.npy',allow_pickle=False);own=np.load(original/'own_predicted_execution.npy',allow_pickle=False)
   detail=dict(choice=0,origin='inherited_auxiliary_PDM_fallback',inherited_fallback=True,guard_changed=False,source=str(original))
  else:
   same=previous is None or np.array_equal(previous_pose,np.load(e.R18/'predictions'/e.C0/previous_token/'trajectory.npy',allow_pickle=False))
   a=e.arrays(original/'candidate_features.npz');gr=e.read(e.R18/'predictions/CoordiWorld-PNC_H3P_staticGuard_v01'/token/'receipt.json')
   geo=e.arrays(e.Path(gr['source_geometry']));native=geo['execution_native'];unsafe=geo['unsafe'];bankrec=old
   ec=None if previous is None else e.pnc.ns['ec_matrix'](native,previous,round((old['currenttimestamp']-previous_ts)/1e5))
   trigger=False if ec is None else not bool((ec[:12][a['valid'][:12]]>0).any())
   live=not same and (bool(old['variant_steps']) or trigger)
   bank_source=original/'candidate_features.npz'
   if live:
    ensure_model();path=row['raw_log']
    if path not in logs:assert e.sha(path)==row['raw_log_sha256'];logs={path:pickle.loads(e.Path(path).read_bytes())}
    h=logs[path][row['index']-3:row['index']+1];assert len(h)==4 and str(h[-1]['token'])==token
    native,a,bankrec,assets,current,brake_slot=infer(pipe,model,h,row['raw_log_sha256'],previous,previous_ts)
    geo=e.frozen_guard(native,h[-1],a['valid']);unsafe=geo['unsafe']
    ec=None if previous is None else e.pnc.ns['ec_matrix'](native,previous,round((old['currenttimestamp']-previous_ts)/1e5))
    np.savez_compressed(dest/'candidate_features.npz',**a);np.savez_compressed(dest/'head_outputs.npz',**model.last_prediction)
    np.savez_compressed(dest/'world_assets.npz',**{k:v for k,v in assets.items() if isinstance(v,np.ndarray)})
    np.savez_compressed(dest/'geometry.npz',execution_native=native,unsafe=unsafe,valid=a['valid'])
    e.write(dest/'current.json',current);e.write(dest/'bank_receipt.json',bankrec)
    bank_source=dest/'candidate_features.npz';live_calls+=1
   chosen,selection=e.simple_select(a,ec,rule,unsafe)
   poses=a['candidates'][chosen];own=native[chosen]
   if np.array_equal(poses,np.load(original/'trajectory.npy',allow_pickle=False)):
    reference=np.load(original/'own_predicted_execution.npy',allow_pickle=False)
    assert np.max(np.abs(own-reference))<=1e-7;own=reference
   origins=e.origin(bankrec,len(a['valid']))
   np.savez_compressed(dest/'selector.npz',ordering=selection['ordering'],valid=selection['proxies']['valid'],unsafe=unsafe,
       EC_proxy=np.full(len(a['valid']),np.nan) if ec is None else ec,**{k:v for k,v in selection['proxies'].items() if k not in ['valid','EC_proxy']})
   detail=dict(choice=chosen,pre_guard_choice=selection['pre_guard_choice'],origin=origins[chosen],inherited_fallback=False,
      guard_changed=selection['guard_changed'],candidate_count=len(a['valid']),valid_candidates=int(a['valid'].sum()),
      bank_source=str(bank_source),bank_sha256=e.sha(bank_source),same_original_previous=same,original_H3_trigger=bool(old['variant_steps']),recomputed_H3_trigger=trigger)
  np.save(dest/'trajectory.npy',poses);np.save(dest/'own_predicted_execution.npy',own)
  rec=dict(status='PASS',at=e.now(),token=token,method=e.S1,Simple_freeze_sha256=freeze,rule=rule,
      history=history,currenttimestamp=old['currenttimestamp'],scored=bool(row['scored']),live_reentry=live,
      changed_from_C0=not np.array_equal(poses,np.load(original/'trajectory.npy',allow_pickle=False)),
      changed_from_C1=not np.array_equal(poses,np.load(e.R18/'predictions'/e.C1/token/'trajectory.npy',allow_pickle=False)),
      trajectory_sha256=e.sha(dest/'trajectory.npy'),execution_sha256=e.sha(dest/'own_predicted_execution.npy'),seconds=time.monotonic()-t0,**detail)
  e.write(receipt,rec);signal.alarm(0)
  records.append(dict(token=token,scored=bool(row['scored']),receipt_sha256=e.sha(receipt)))
  previous=own.copy();previous_pose=poses.copy();previous_ts=old['currenttimestamp'];previous_token=token
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<6*2**30,'RAM resource contract exceeded'
 e.write(e.RUN/'configs/S1_OWN_SELECTION_SEAL.json',dict(at=e.now(),status='PASS',inputs=2919,formal=1336,records=records,
     Simple_freeze_sha256=freeze,source_sha256=e.sha(__file__),live_reentries=live_calls,world_calls=world_calls,seconds=time.monotonic()-started,before_own_native_evaluation=True))
 e.write(e.RUN/'queue/OWN_INFERENCE.json',dict(at=e.now(),status='COMPLETE',done=2919,formal=1336,live_reentries=live_calls,seconds=time.monotonic()-started))
 print('OWN_SELECTION_COMPLETE',2919,live_calls)

def evaluate():
 e.verify_implementation()
 held=lock('OWN_EVALUATION.lock');seal=e.read(e.RUN/'configs/S1_OWN_SELECTION_SEAL.json');assert seal['status']=='PASS'
 from conditional_comparison import Scorer
 import common_runtime as cr
 import pandas as pd,lzma
 scorer=Scorer();rows=e.cohort();start=time.monotonic();calls=reuses=0
 sealed={r['token']:r for r in seal['records']}
 for i,row in enumerate(rows):
  token=row['token'];pred=e.RUN/'predictions'/e.S1/token;out=e.RUN/'results'/e.S1/token
  assert e.sha(pred/'receipt.json')==sealed[token]['receipt_sha256']
  if (out/'receipt.json').exists():
   rec=e.read(out/'receipt.json');assert rec['selection_seal_sha256']==e.sha(e.RUN/'configs/S1_OWN_SELECTION_SEAL.json');continue
  poses=np.load(pred/'trajectory.npy',allow_pickle=False);expected=np.load(pred/'own_predicted_execution.npy',allow_pickle=False)
  comp,raw,states,receipt=scorer.one(row,poses,expected,out)
  table=pd.read_pickle(out/'score_raw.pkl')
  cache=pickle.loads(lzma.decompress((e.R18/'data/metric_cache'/row['log']/'unknown'/token/'metric_cache.pkl').read_bytes()))
  table['token']=token;table['valid']=True;table['log_name']=row['log'];table['drive']=row['drive'];table['frame_type']=cache.scene_type;table['start_time']=cache.timepoint.time_s;table['ego_simulated_states']=[states];table['scored']=row['scored']
  table.to_pickle(out/'score_raw.pkl');calls+=receipt['reused_from'] is None;reuses+=receipt['reused_from'] is not None
  e.write(out/'receipt.json',dict(at=e.now(),status='PASS',selection_seal_sha256=e.sha(e.RUN/'configs/S1_OWN_SELECTION_SEAL.json'),
       score_sha256=e.sha(out/'score_raw.pkl'),execution_sha256=e.sha(out/'execution.npy'),trajectory_sha256=e.sha(out/'trajectory.npy'),rawNC=raw,reused_from=receipt['reused_from']))
  e.write(e.RUN/'queue/OWN_EVALUATION.json',dict(at=e.now(),status='RUNNING',pid=os.getpid(),done=i+1,total=2919,new_calls=calls,reuses=reuses,seconds=time.monotonic()-start))
 # The original official aggregation functions and their scene-equal denominator.
 cr.N=e.RUN;dest=e.RUN/'analysis/S1_own';dest.mkdir(parents=True,exist_ok=True);summary=cr.aggregate(e.S1,rows,dest)
 completed=[e.read(e.RUN/'results'/e.S1/r['token']/'receipt.json') for r in rows]
 calls=sum(r['reused_from'] is None for r in completed);reuses=len(completed)-calls
 e.write(e.RUN/'evidence/S1_OWN_COMPLETE.json',dict(at=e.now(),status='PASS',summary=summary,inputs=2919,formal=1336,new_native_calls=calls,reused_native_labels=reuses,seconds=time.monotonic()-start,
     scored_tokens_sha256=e.sha(dest/'scored_tokens.csv'),selection_seal_sha256=e.sha(e.RUN/'configs/S1_OWN_SELECTION_SEAL.json'),source_sha256=e.sha(__file__)))
 e.write(e.RUN/'queue/OWN_EVALUATION.json',dict(at=e.now(),status='COMPLETE',done=2919,total=2919,new_calls=calls,reuses=reuses))
 print('S1_OWN_COMPLETE',json.dumps(summary))

if __name__=='__main__':
 import sys,json
 try:
  if sys.argv[1:]==['--generate']:generate()
  elif sys.argv[1:]==['--evaluate']:evaluate()
  else:raise ValueError('explicit stage required')
 except Exception as error:
  e.write(e.RUN/'failures'/f'own_{os.getpid()}_{time.time_ns()}.json',dict(at=e.now(),error=repr(error),traceback=traceback.format_exc()))
  raise
