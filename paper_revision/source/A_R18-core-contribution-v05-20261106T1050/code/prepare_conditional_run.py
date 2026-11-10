"""Validate saved identities, freeze four-arm choices and the implementation.
No model inference or new native labels. Invoked once under submission lock.
"""
import fcntl,json,sys,time
import experiment_common as e
from conditional_comparison import prepare
def main():
 p=e.RUN/'locks/SUBMISSION.lock';p.parent.mkdir(exist_ok=True);lock=p.open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (e.RUN/'configs/IMPLEMENTATION_FREEZE_v01.json').exists()
 start=time.monotonic();cohort=e.cohort();pins={};sizes=0
 expected=e.read(e.RUN/'evidence/ENTRYPOINT_AUDIT.json')
 for name,k in [('repair_live_history_v01.py','C1_source_sha256'),('current_static_geometry_v01.py','C1_guard_sha256')]:assert e.sha(e.R18/'code'/name)==expected[k]
 assert e.sha(e.RUN/'configs/SIMPLE_FINAL_FREEZE_v01.json')=='2d06e8ab413c8d42f32a2ee7fca0c2b24ad1c0d9a2735ea2463173f45a102c28'
 assert e.sha(e.RUN/'configs/CORE_CONTRIBUTION_FREEZE_v01.json')=='dba76f16991ef2a22c10239c2b8909e0b6c0df89433be23257fe99843c1cf098'
 for row in cohort:
  t=row['token'];cache=e.R18/'data/metric_cache'/row['log']/'unknown'/t/'metric_cache.pkl';rec=e.read(e.R18/'results'/e.C0/t/'receipt.json')
  digest=e.sha(cache);assert rec['metric_cache_sha256']==digest,(t,'old cache byte identity')
  pins[t]={'path':str(cache),'sha256':digest,'bytes':cache.stat().st_size};sizes+=cache.stat().st_size
  for method in [e.C0,e.C1]:
   d=e.R18/'results'/method/t;assert all((d/n).is_file() for n in ['trajectory.npy','execution.npy','score_raw.pkl','receipt.json'])
 e.write(e.RUN/'configs/CACHE_IDENTITY_MANIFEST.json',pins)
 print('SAVED_CACHE_IDENTITIES_PASS',len(pins),sizes,flush=True)
 prepare()
 frozen=e.read(e.RUN/'configs/CONDITIONAL_CHOICES_FROZEN_v01.json');demand=[]
 import numpy as np
 for row in frozen['rows']:
  bank,rec=e.c1_bank(row['row']);seen=set()
  for arm,j in row['choices'].items():
   key=row['trajectory_ids'][arm]
   if key in seen:continue
   seen.add(key);trajectory=bank['a']['candidates'][j];reuse=None
   for method in [e.C1,e.C0,'PDM-Closed']:
    p=e.R18/'results'/method/row['token']/'trajectory.npy'
    if p.is_file() and np.array_equal(np.load(p,allow_pickle=False),trajectory):reuse=str(p.parent);break
   demand.append({'token':row['token'],'drive':row['drive'],'trajectory_id':key,'existing_exact_trajectory':reuse,'new_native_required':reuse is None})
 e.write(e.RUN/'configs/CONDITIONAL_LABEL_DEMAND.json',{'at':e.now(),'unique':len(demand),'new_native_required':sum(x['new_native_required'] for x in demand),'rows':demand,'outcomes_read':False})
 files={}
 for manifest in ['SOURCE_MANIFEST.json','SIMPLE_SOURCE_MANIFEST.json','OFFICIAL_AGGREGATION_SOURCES.json']:
  for x in e.read(e.RUN/'evidence'/manifest):
   if 'sha256' not in x:continue
   path=e.Path(x['path']);assert e.sha(path)==x['sha256'],(path,'historical source changed');files[str(path)]=x['sha256']
 for folder,names in [('code',['simple_rules.py','experiment_common.py','conditional_comparison.py','run_simple_own.py']),('configs',['CORE_CONTRIBUTION_FREEZE_v01.json','SIMPLE_FINAL_FREEZE_v01.json','CONDITIONAL_CHOICES_FROZEN_v01.json','CACHE_IDENTITY_MANIFEST.json','CONDITIONAL_LABEL_DEMAND.json'])]:
  for n in names:files[str(e.RUN/folder/n)]=e.sha(e.RUN/folder/n)
 baseline=e.R18/'analysis/static_guard_live_history_v01'/e.C1/'scored_tokens.csv'
 assert e.sha(baseline)=='6c12cf12077ac0ed36a551a165781351d2cf3397b18edf7521c428719615188c'
 files[str(baseline)]=e.sha(baseline)
 e.write(e.RUN/'configs/IMPLEMENTATION_FREEZE_v01.json',{'at':e.now(),'before_new_v05_evaluation':True,'path':'B','new_models':0,'files':files,'historical_cache_identity':'all2919 matched label-time SHA in original M0 receipt','shared_proposer':'historical exact C1 numerical body; final selector Simple A; resource cache adapter retains existing equivalence','only_one_new_policy':e.S1})
 e.verify_implementation()
 out={'at':e.now(),'status':'PASS','scored':1336,'causal_inputs':2919,'drives':20,'cache_bytes_verified':sizes,'sources_pinned':len(files),'unique_selected_trajectories':len(demand),'new_native_required':sum(x['new_native_required'] for x in demand),'seconds':time.monotonic()-start,'new_model_calls':0,'new_native_calls':0,'implementation_freeze_sha256':e.sha(e.RUN/'configs/IMPLEMENTATION_FREEZE_v01.json')}
 e.write(e.RUN/'evidence/CONDITIONAL_PREFLIGHT.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
