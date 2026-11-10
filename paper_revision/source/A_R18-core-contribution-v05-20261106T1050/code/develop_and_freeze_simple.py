"""Fixed TRAIN comparator selection; writes choices before reading NR labels."""
import json,pickle,time,sys
import experiment_common as e
import numpy as np
from simple_rules import RULES,select

def prepare():
 cfg=e.read(e.RUN/'configs/CORE_CONTRIBUTION_FREEZE_v01.json')
 assert e.sha(e.RUN/'code/simple_rules.py')==cfg['simple_source_sha256']
 roster_path=e.V04/'configs/PAIRED_TARGET_FREEZE_v01.json'
 assert e.sha(roster_path)==cfg['training_roster_freeze_sha256']
 roster=e.read(roster_path)['roster'];loaded={};choices=[]
 for row in roster:
  t=row['token'];d=e.V03/'data/current_M0'/t;a=e.arrays(d/'candidate_features.npz');rec=e.read(d/'receipt.json')
  assert e.sha(d/'candidate_features.npz')==row['input_sha256'][str(d/'candidate_features.npz')]
  raw=rec['raw_log']
  if raw not in loaded:
   assert e.sha(raw)==row['history']['raw_log_sha256'];loaded={raw:pickle.loads(e.Path(raw).read_bytes())}
  native=np.load(d/'all_predicted_executions.npy',allow_pickle=False)
  g=e.frozen_guard(native,loaded[raw][rec['current_index']],a['valid'])
  for rule in RULES:
   un,detail=select(a,a['EC'],rule);guarded,gd=select(a,a['EC'],rule,g['unsafe'])
   assert un in row['valid_indices'] and guarded in row['valid_indices']
   # A hostile/unrelated head payload cannot affect the Simple selection.
   a2=dict(a,eligible=np.zeros(len(a['valid']),bool),CW_head=np.full(len(a['valid']),1e9))
   assert select(a2,a['EC'],rule,g['unsafe'])[0]==guarded
   choices.append(dict(token=t,drive=row['drive'],fold=row['fold'],rule=rule,S0=un,S1=guarded,
      candidate_bank_sha256=e.sha(d/'candidate_features.npz'),guard_changed=gd['guard_changed'],valid_candidates=len(row['valid_indices']),
      S0_origin=str(a['candidate_origin'][un]),S1_origin=str(a['candidate_origin'][guarded]),
      unknown_features_at_S1=int(gd['proxies']['support_missing_count'][guarded])))
 e.write(e.RUN/'configs/TRAIN_SIMPLE_CHOICES_FROZEN.json',dict(at=e.now(),before_new_label_reads=True,core_freeze_sha256=e.sha(e.RUN/'configs/CORE_CONTRIBUTION_FREEZE_v01.json'),roster_sha256=e.sha(roster_path),rows=choices))
 print('TRAIN_CHOICES_FROZEN',len(choices))

def evaluate():
 assert e.read(e.V04/'QUEUE_STATUS.json')['status']=='PROBE_COMPLETE','Finish the entire unchanged v04 frozen block first'
 frozen=e.read(e.RUN/'configs/TRAIN_SIMPLE_CHOICES_FROZEN.json');cfg=e.read(e.RUN/'configs/CORE_CONTRIBUTION_FREEZE_v01.json')
 assert frozen['core_freeze_sha256']==e.sha(e.RUN/'configs/CORE_CONTRIBUTION_FREEZE_v01.json')
 assert e.sha(e.RUN/'code/simple_rules.py')==cfg['simple_source_sha256']
 rows=[]
 for r in frozen['rows']:
  d=e.V04/'data/probe'/r['token'];receipt=e.read(d/'receipt.json');assert receipt['status']=='PASS'
  assert e.sha(d/'paired_labels.npz')==receipt['paired_labels_sha256'];z=e.arrays(d/'paired_labels.npz')
  assert receipt['two_arm_execution_max_error']<=1e-7
  for arm in ['S0','S1']:
   j=r[arm];c=z['components'][1,j];raw=z['raw_components'][1,j];ec=float(z['EC'][1,j]);assert np.isfinite(c).all() and np.isfinite(ec)
   u=float(c[:4].prod()*(5*c[4]+5*c[5]+2*c[6]+2*c[7]+2*ec)/16)
   rows.append(dict(token=r['token'],drive=r['drive'],rule=r['rule'],arm=arm,candidate=j,origin=r[arm+'_origin'],Score=100*u,
       rawNC=float(raw[0]),NC=float(c[0]),DAC=float(c[1]),DDC=float(c[2]),TLC=float(c[3]),EP=float(c[4]),TTC=float(c[5]),LK=float(c[6]),HC=float(c[7]),EC=ec,
       guard_changed=r['guard_changed'],unknown_features=r['unknown_features_at_S1']))
 summary=[]
 for rule in RULES:
  for arm in ['S0','S1']:
   rr=[r for r in rows if r['rule']==rule and r['arm']==arm]
   summary.append(dict(rule=rule,arm=arm,scenes=len(rr),drives=len({r['drive'] for r in rr}),scene_mean_Score=float(np.mean([r['Score'] for r in rr])),
      events={k:sum(r[k]<1 for r in rr) for k in ['rawNC','NC','DAC','DDC','TLC','TTC','LK','HC','EC']},progress=float(np.mean([r['EP'] for r in rr]))))
 primary={r['rule']:r['scene_mean_Score'] for r in summary if r['arm']=='S1'}
 chosen=RULES[1] if primary[RULES[1]]>primary[RULES[0]]+1e-10 else RULES[0]
 e.csvwrite(e.RUN/'analysis/TRAIN_SIMPLE_DEVELOPMENT.csv',rows)
 e.write(e.RUN/'configs/SIMPLE_FINAL_FREEZE_v01.json',dict(at=e.now(),rule=chosen,selection_protocol=cfg['Simple_selection'],
    core_freeze_sha256=e.sha(e.RUN/'configs/CORE_CONTRIBUTION_FREEZE_v01.json'),source_sha256=e.sha(e.RUN/'code/simple_rules.py'),
    TRAIN_choices_sha256=e.sha(e.RUN/'configs/TRAIN_SIMPLE_CHOICES_FROZEN.json'),summary=summary,
    result_sha256=e.sha(e.RUN/'analysis/TRAIN_SIMPLE_DEVELOPMENT.csv'),before_1336_v05_results=True,new_models=0,classification='comparator development, exposed upstream, not independent confirmation'))
 print(json.dumps(dict(chosen=chosen,summary=summary)))
if __name__=='__main__':
 if sys.argv[1:]==['--prepare']:prepare()
 elif sys.argv[1:]==['--evaluate']:evaluate()
 else:raise ValueError('Explicit operation required')
