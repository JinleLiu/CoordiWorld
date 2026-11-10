"""Four arms on exact saved C1 banks/history; unique selected labels only."""
import argparse,fcntl,lzma,pickle,time,traceback
import experiment_common as e
import numpy as np
METRICS=['no_at_fault_collisions','drivable_area_compliance','driving_direction_compliance','traffic_light_compliance','ego_progress','time_to_collision_within_bound','lane_keeping','history_comfort']

def prepare():
 frozen=e.read(e.RUN/'configs/SIMPLE_FINAL_FREEZE_v01.json');rule=frozen['rule']
 assert frozen['source_sha256']==e.sha(e.RUN/'code/simple_rules.py')
 cw=e.original_selector();rows=[]
 for row in e.cohort():
  if not row['scored']:continue
  bank,rec=e.c1_bank(row);assert bank is not None
  a=bank['a'];c0,c1,eligible,scores=cw(a,bank['heads'],bank['ec_proxy'],bank['bankrec'],bank['unsafe'])
  assert c1==rec['choice'],(row['token'],'historical C1 selector mismatch')
  assert np.array_equal(eligible,bank['selector']['eligible'])
  assert np.max(np.abs(scores-bank['selector']['scores']))<1e-12
  s0,_=e.simple_select(a,bank['ec_proxy'],rule);s1,sd=e.simple_select(a,bank['ec_proxy'],rule,bank['unsafe'])
  origins=e.origin(bank['bankrec'],len(a['valid']));choices={'C0':c0,'C1':c1,'S0':s0,'S1':s1}
  rows.append(dict(token=row['token'],drive=row['drive'],log=row['log'],block=row['block'],history_owner='C1',
    currenttimestamp=rec['currenttimestamp'],previous_token=rec.get('previous'),previous_timestamp=rec.get('own_previous_timestamp'),
    bank_path=str(bank['bank_path']),bank_sha256=bank['bank_sha256'],choices=choices,
    origins={arm:origins[j] for arm,j in choices.items()},valid_candidates=int(a['valid'].sum()),CW_eligible=int(eligible.sum()),
    Simple_valid=int(sd['proxies']['valid'].sum()),trajectory_ids={arm:e.hashlib.sha256(np.ascontiguousarray(a['candidates'][j]).tobytes()).hexdigest() for arm,j in choices.items()},
    C_guard_changed=c1!=c0,S_guard_changed=s1!=s0,live_C1_bank=rec['live_reentry'],row=row))
 assert len(rows)==1336 and len({r['drive'] for r in rows})==20
 p=e.RUN/'configs/CONDITIONAL_CHOICES_FROZEN_v01.json';assert not p.exists()
 e.write(p,dict(at=e.now(),before_conditional_label_reads=True,Simple_freeze_sha256=e.sha(e.RUN/'configs/SIMPLE_FINAL_FREEZE_v01.json'),
    source_sha256=e.sha(__file__),rows=rows,scenes=1336,drives=20,unique_selected_trajectories=sum(len(set(r['trajectory_ids'].values())) for r in rows),
    C1_saved_choice_parity='all1336 exact; head eligibility exact; scores atol1e-12',new_model_calls=0))
 print('CONDITIONAL_CHOICES_FROZEN',len(rows),sum(len(set(r['trajectory_ids'].values())) for r in rows))

class Scorer:
 def __init__(self):
  import align_train_benchmark as b
  import ast
  self.b=b;self.owner=b.OfficialComponentScorer(b.pnc.m.ROOT/'data/maps',e.RUN/'unused_metric_cache','non_reactive')
  source=next((b.pnc.m.PEER/'source').rglob('revision_v5_twentythird_native.py'))
  node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='Capture')
  ns={'METRICS':METRICS};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
  self.capture=ns['Capture'](self.owner.scorer)
  self.raw_reference_path=e.R18/'analysis/selected_raw_nc_calibration_v01/per_scene.csv'
  assert e.sha(self.raw_reference_path)=='f8d66d956d702983864f86a00a220d03f746427aff4a5a684d3396210d89837f'
  self.raw_reference={r['token']:float(r['raw_NC']) for r in e.csv.DictReader(self.raw_reference_path.open())}
  self.cache_identities={}
 def cache_identity(self,row):
  token=row['token']
  if token not in self.cache_identities:
   path=e.R18/'data/metric_cache'/row['log']/'unknown'/token/'metric_cache.pkl'
   historical=e.read(e.R18/'results'/e.C0/token/'receipt.json')
   actual=e.sha(path)
   assert historical['metric_cache_sha256']==actual,(token,'historical metric-cache identity changed')
   self.cache_identities[token]=(path,actual)
  return self.cache_identities[token]
 def one(self,row,poses,predicted,output,allow_reuse=True):
  """Native labels exclude adjacent EC, which is separately bound to history."""
  output.mkdir(parents=True,exist_ok=True);token=row['token'];reused=None;raw=None;raw_nc=None
  cache_path,cache_sha=self.cache_identity(row)
  if allow_reuse:
   sources=[e.R18/'results'/method/token for method in [e.C1,e.C0,'PDM-Closed']]
   sources+=sorted((e.RUN/'data/conditional'/token).glob('*'))
   for src in sources:
    if src==output:continue
    if not (src/'score_raw.pkl').is_file() or not (src/'trajectory.npy').is_file():continue
    oldposes=np.load(src/'trajectory.npy',allow_pickle=False)
    if np.array_equal(oldposes,poses):
     if (src/'native_label_receipt.json').is_file():
      oldrec=e.read(src/'native_label_receipt.json')
      assert oldrec['metric_cache_sha256']==cache_sha and oldrec['input_log_sha256']==row['raw_log_sha256']
      assert oldrec['evaluation_source_sha256']==e.sha(e.R18/'code/common_runtime.py')
      assert oldrec['protocol']=='non-reactive official pdm_score, human_filter=true, original pair progress normalization'
     states=np.load(src/'execution.npy',allow_pickle=False)
     assert float(np.max(np.abs(states-predicted)))<=1e-7,(token,'reused execution identity')
     table=__import__('pandas').read_pickle(src/'score_raw.pkl');reused=src
     if (src/'native_label_receipt.json').is_file():raw_nc=e.read(src/'native_label_receipt.json')['rawNC']
     break
  if reused is None:
   cache=pickle.loads(lzma.decompress(cache_path.read_bytes()))
   comp,raw,states,table=self.b.score_candidate(self.owner,self.capture,cache,poses)
   assert float(np.max(np.abs(states-predicted)))<=1e-7,(token,'new execution identity')
  else:comp=np.array([float(table.iloc[0][k]) for k in METRICS])
  assert np.isfinite(comp).all()
  if raw is not None:raw_nc=float(raw[0])
  raw_source='new official pre-filter Capture' if raw is not None else ('prior exact native label receipt' if raw_nc is not None else None)
  if raw_nc is None and token in self.raw_reference and np.array_equal(poses,np.load(e.R18/'predictions'/e.C0/token/'trajectory.npy',allow_pickle=False)):
   raw_nc=self.raw_reference[token];raw_source=str(self.raw_reference_path)+'; exact original M0 trajectory'
  table.to_pickle(output/'score_raw.pkl');np.save(output/'execution.npy',states);np.save(output/'trajectory.npy',poses)
  # A reused filtered result alone never licenses an invented raw NC value.
  rec=dict(at=e.now(),token=token,components=comp.tolist(),rawNC=raw_nc,rawNC_source=raw_source,reused_from=None if reused is None else str(reused),
      trajectory_array_sha256=e.hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest(),
      source_score_sha256=e.sha((reused or output)/'score_raw.pkl'),source_execution_sha256=e.sha((reused or output)/'execution.npy'),
      input_log_sha256=row['raw_log_sha256'],cache_path=str(cache_path),metric_cache_sha256=cache_sha,
      protocol='non-reactive official pdm_score, human_filter=true, original pair progress normalization',
      evaluation_source_sha256=e.sha(e.R18/'code/common_runtime.py'),conditional_EC_included=False)
  e.write(output/'native_label_receipt.json',rec)
  return comp,raw_nc,states,rec

def evaluate():
 e.verify_implementation()
 frozen=e.read(e.RUN/'configs/CONDITIONAL_CHOICES_FROZEN_v01.json')
 assert frozen['Simple_freeze_sha256']==e.sha(e.RUN/'configs/SIMPLE_FINAL_FREEZE_v01.json')
 assert frozen['source_sha256']==e.sha(__file__)
 p=e.RUN/'locks/CONDITIONAL_LIFETIME.lock';p.parent.mkdir(exist_ok=True);lock=p.open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 scorer=Scorer();rows=[];start=time.monotonic();native_calls=0;reuses=0
 for index,r in enumerate(frozen['rows']):
  bank,old=e.c1_bank(r['row']);assert e.sha(r['bank_path'])==r['bank_sha256']
  previous=None if r['previous_token'] is None else np.load(e.R18/'results'/e.C1/r['previous_token']/'execution.npy',allow_pickle=False)
  labels={}
  out=e.RUN/'data/conditional'/r['token'];out.mkdir(parents=True,exist_ok=True)
  existing=out/'scene_receipt.json'
  if existing.exists():
   rr=e.read(existing);assert rr['freeze_sha256']==e.sha(e.RUN/'configs/CONDITIONAL_CHOICES_FROZEN_v01.json');rows.extend(rr['rows']);continue
  for arm,j in r['choices'].items():
   key=r['trajectory_ids'][arm]
   if key not in labels:
    c,raw,states,receipt=scorer.one(r['row'],bank['a']['candidates'][j],bank['native'][j],out/key)
    native_calls+=receipt['reused_from'] is None;reuses+=receipt['reused_from'] is not None
    ec=None if previous is None else float(e.pnc.ns['ec_matrix'](states[None],previous,round((r['currenttimestamp']-r['previous_timestamp'])/1e5))[0])
    labels[key]=(c,raw,ec,receipt)
  scene_rows=[]
  for arm,j in r['choices'].items():
   c,raw,ec,receipt=labels[r['trajectory_ids'][arm]]
   u=None if ec is None else float(c[:4].prod()*(5*c[4]+5*c[5]+2*c[6]+2*c[7]+2*ec)/16)
   scene_rows.append(dict(token=r['token'],drive=r['drive'],arm=arm,history_owner='C1',candidate=j,candidate_origin=r['origins'][arm],
      trajectory_id=r['trajectory_ids'][arm],Score=None if u is None else 100*u,rawNC=raw,NC=float(c[0]),DAC=float(c[1]),DDC=float(c[2]),TLC=float(c[3]),
      EP=float(c[4]),TTC=float(c[5]),LK=float(c[6]),HC=float(c[7]),EC=ec,CW_eligible=r['CW_eligible'],Simple_valid=r['Simple_valid'],
      native_reused=receipt['reused_from'] is not None,previous_token=r['previous_token'],bank_sha256=r['bank_sha256']))
  rows.extend(scene_rows)
  e.write(existing,dict(status='PASS',at=e.now(),freeze_sha256=e.sha(e.RUN/'configs/CONDITIONAL_CHOICES_FROZEN_v01.json'),rows=scene_rows))
  e.write(e.RUN/'queue/CONDITIONAL.json',dict(at=e.now(),status='RUNNING',pid=__import__('os').getpid(),done=index+1,total=1336,new_native_calls=native_calls,reused_native_labels=reuses,seconds=time.monotonic()-start))
 # Count sealed unique label receipts, including successes restored after a process interruption.
 label_receipts=[e.read(p) for p in (e.RUN/'data/conditional').glob('*/*/native_label_receipt.json')]
 native_calls=sum(r['reused_from'] is None for r in label_receipts);reuses=len(label_receipts)-native_calls
 e.csvwrite(e.RUN/'analysis/FOUR_ARM_CONDITIONAL.csv',rows)
 e.write(e.RUN/'evidence/CONDITIONAL_COMPLETE.json',dict(at=e.now(),status='PASS',scenes=1336,rows=len(rows),seconds=time.monotonic()-start,
    result_sha256=e.sha(e.RUN/'analysis/FOUR_ARM_CONDITIONAL.csv'),new_native_calls=native_calls,reused_native_labels=reuses,history_owner='C1',not_own_history=True))
 e.write(e.RUN/'queue/CONDITIONAL.json',dict(at=e.now(),status='COMPLETE',done=1336,total=1336,new_native_calls=native_calls,reused_native_labels=reuses))
 print('CONDITIONAL_COMPLETE',len(rows),native_calls,reuses)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--evaluate',action='store_true');a=ap.parse_args()
 if a.prepare:prepare()
 elif a.evaluate:evaluate()
 else:ap.error('explicit action required')
