# coding: utf-8
"""Frozen PNC_H3P on original four-frame logs; policy-owned causal plan buffer.
Numerical functions are the deployed B23/R14 bodies, bound to preserved assets.
This process never imports the common evaluator or reads outcome labels.
"""
import os,sys,ast,json,hashlib,time,copy,argparse,traceback
from pathlib import Path
R=Path('${CW_ARCHIVE_ROOT}');N=R/'A_R18-research-20261031T1047';OLD=R/'A_R15-research-20261030T0013'
sys.path[:0]=[str(OLD/'source'),str(N/'deps')]
import minimal_runtime as m
import numpy as np,torch,xgboost as xgb
from coordiworld.revision_v5.compiled_solve_controller import predict_vector_controller_execution
from revision_v5_seventeenth_math import physical_progress_proxy

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n');q.replace(p)
bindings=[];ns=dict(m.__dict__);ns.update(physical_progress_proxy=physical_progress_proxy,predict_vector_controller_execution=predict_vector_controller_execution)
def bind(filename,names,source=None):
 p=source or next((m.PEER/'source').rglob(filename));s=p.read_text();nodes=[v for v in ast.parse(s).body if isinstance(v,(ast.FunctionDef,ast.ClassDef))and(v.name in names if names else v.name!='main')]
 if names:assert {v.name for v in nodes}==set(names),(filename,names)
 bindings.append(dict(source=str(p),sha256=sha(p),symbols=[v.name for v in nodes],ast_sha256=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()))
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
bind('revision_v5_twentythird_factory.py',None)
bind('revision_v5_twentythird_features.py',['p_context','add_candidate','replace_from_appended'])
bind('revision_v5_twentysecond_primary.py',['project'])
bind('revision_v5_twentythird_policy.py',['PNC'])
bind('revision_v5_twentythird_qualification.py',['choice'])
bind('revision_v5_twentythird_progress_readout.py',['select']);ns['hp_select']=ns['select']
bind('revision_v5_twentythird_null_probe.py',['current_profile_candidate'])
bind('revision_v5_twentythird_spatial_probe.py',['spatial_dynamic'])
bind('revision_v5_twentysecond_pool_probe.py',['brake'])
bind('revision_v5_twentieth_pool_probe.py',['slot'])
bind('revision_v5_twentythird_timing_active.py',['infer'],N/'code/peer_reference/revision_v5_twentythird_timing_active.py')

class FrozenPNC(ns['PNC']):
 def __init__(self):
  self.models={};self.bindings={}
  for h in ['q','qp','qttc','qlk','dac_event','physical_nc']:
   p=m.PEER/'models/PNC'/h/'model.ubj';booster=xgb.Booster();booster.load_model(p);booster.set_param({'nthread':1});self.models[h]=booster;self.bindings[h]={'path':str(p),'sha256':sha(p)}
 def predict(self,a,ec=None):
  p=super().predict(a,ec);self.last_prediction=p;return p

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--block',type=int,default=0);ap.add_argument('--limit',type=int,default=3);ap.add_argument('--run',required=True);ap.add_argument('--reuse-successful',action='store_true');args=ap.parse_args()
 bp=N/'data'/f'cohort_block{args.block:03}.json';rows=json.load(open(bp))['records'];rows=rows[:args.limit]if args.limit else rows
 out=N/'model_runs'/args.run;out.mkdir(parents=True,exist_ok=False);start=time.time();deadline=min(start+7200,1793501252.)
 write(out/'plan.json',dict(method='CoordiWorld-PNC_H3P',source_sha256=sha(__file__),cohort_sha256=sha(bp),tokens=[r['token']for r in rows],deadline_unix=deadline,world='factual predictor plus common physical rails',previous_plan='own predicted execution only; reset at metadata segment_start',four_current_frames=True,read_outcomes=False,training=False,source_bindings=bindings))
 pipe=m.Pipeline();model=FrozenPNC();expected=json.load(open(N/'code/pnc_expected_bindings.json'));assert pipe.checkpoint_sha256==expected['world'];assert {k:v['sha256']for k,v in model.bindings.items()}==expected['heads'];write(out/'weight_binding.json',dict(world_checkpoint_sha256=pipe.checkpoint_sha256,heads=model.bindings,runtime_source_bindings=m.BINDINGS+bindings,model_load_seconds=time.time()-start))
 previous=None;previous_ts=None;loaded={};records=[]
 for row in rows:
  if time.time()>deadline:raise TimeoutError('fixed model block deadline')
  token=row['token'];dest=N/'predictions/CoordiWorld-PNC_H3P'/token
  if row.get('segment_start',False):previous=None;previous_ts=None
  if dest.exists():
   rec=json.load(open(dest/'receipt.json'));assert args.reuse_successful and rec['status']=='PASS'and rec['own_previous_timestamp']==previous_ts
   assert sha(dest/'trajectory.npy')==rec['trajectory_sha256'];previous=np.load(dest/'own_predicted_execution.npy');previous_ts=rec['currenttimestamp'];records.append(rec|{'reused':True});continue
  dest.mkdir(parents=True,exist_ok=False)
  try:
   if row.get('segment_start',False):previous=None;previous_ts=None
   path=row['raw_log']
   if path not in loaded:
    assert sha(path)==row['raw_log_sha256'];loaded={path:m.load_log(path)}
   i=row['index'];h=loaded[path][i-3:i+1];assert str(h[-1]['token'])==token and len(h)==4
   # A log/segment boundary cannot inherit a plan from another drive.
   if previous_ts is not None and not(0<int(h[-1]['timestamp'])-previous_ts<550000):previous=None;previous_ts=None
   native,a,rec=ns['infer'](pipe,model,h,row['raw_log_sha256'],previous,previous_ts)
   poses=np.asarray(a['candidates'][rec['choice']]);assert poses.shape==(8,3)and np.isfinite(poses).all()
   np.save(dest/'trajectory.npy',poses);np.save(dest/'own_predicted_execution.npy',native);np.savez_compressed(dest/'candidate_features.npz',**a);np.savez_compressed(dest/'head_outputs.npz',**model.last_prediction)
   rec.update(status='PASS',token=token,trajectory_sha256=sha(dest/'trajectory.npy'),own_previous_timestamp=previous_ts,coordinate='current rear-axle local x/y/yaw, 0.5 to 4.0 seconds',scored=row['scored'])
   write(dest/'receipt.json',rec);records.append(rec);previous=native;previous_ts=int(h[-1]['timestamp'])
  except Exception as e:
   write(dest/'receipt.json',dict(status='FAILED',token=token,error=repr(e),traceback=traceback.format_exc()));raise
  write(out/'live.json',dict(done=len(records),total=len(rows),last=token,seconds=time.time()-start))
 write(out/'result.json',dict(status='PASS',count=len(records),wall_seconds=time.time()-start,records=records));print('PNC_DONE',len(records),flush=True)
if __name__=='__main__':main()
