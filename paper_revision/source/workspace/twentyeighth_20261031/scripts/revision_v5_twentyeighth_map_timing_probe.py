"""Two fixed slow timings on the already observed static-clear8m entry.

Same source state and frozen world/heads; development diagnostic, no driving.
The hypothesis is time separation from the predicted rear vehicle; risk gates
are unchanged. No future CARLA trajectory enters generation or selection.
"""
import copy,json,time
from pathlib import Path
import numpy as np
from revision_v5_twentyeighth_policy import BridgePolicy,array_history
from revision_v5_twentyeighth_map_supply import MapSupply,CONFIG,local_execution,separation
from revision_v5_twentyeighth_polygon_cache import install as polygons
from revision_v5_twentyeighth_prepared_map import install as prepared
from revision_v5_twentyeighth_short_path import install as shortpath
B=Path('<CW_WORKSPACE>/twentyeighth_20261031');O=B/'map_supply/timing_probe_v01';O.mkdir(parents=True,exist_ok=True);D=B/'bench2drive/episodes/seed17/3514/CoordiWorld_BaseEC/attempt1/trace';row=json.loads((D/'policy/plans.jsonl').read_text().splitlines()[0]);hh=[json.loads(z)for z in open(D/'observed_history.jsonl')];history=[z for z in hh if z['timestamp']<=row['timestamp_us']][-4:];town=json.load(open(D/'agent_metadata.json'))['map_name'].split('/')[-1]
(O/'REGISTRATION.json').write_text(json.dumps(dict(hypothesis='Slower lateral entry may remain behind a currently moving rear car until it passes; same spatial curve and strict original learned risk qualification',source_replan=0,speed_caps_mps=[.8,1.2],join_m=8.,training=False,future_observations=False,continue_if='At least one true positive-forward candidate is risk-qualified and selected with unchanged heads; then test actual driving separately'),indent=2)+'\n')
polygons();prepared(False);shortpath();p=BridgePolicy(json.load(open(B/f'bridge/maps/{town}.json')),'BaseEC',O/'policy','controlled_v02');p.plan(history);saved=np.load(O/'policy/plan_00000.npz');p.prior=None;p.prior_ts=None;fac=MapSupply(p.map);base=p.base_inputs
def inputs(pipe,h,source):
 a,asset,native,current,receipt=base(pipe,h,source);details=[]
 for cap in [.8,1.2]:
  old=CONFIG['speed_max']
  try:CONFIG['speed_max']=cap;proposals,_=fac.generate(h[-1])
  finally:CONFIG['speed_max']=old
  ca,meta,_=proposals[0];meta=dict(meta,speed_cap_mps=cap,admitted=False)
  if ca is not None:
   _,ex,_=p.execution(h,ca[None],np.ones(1,bool));sat,actor=separation(local_execution(ex[0],h[-1]),h[-1]);meta.update(execution_staticSAT=sat,actor=actor,forward_m=float(local_execution(ex[0],h[-1])[-1,0]))
   if sat>=.1:a,asset,native,ar=p.add_candidate(pipe,h,source,a,asset,current,ca);meta.update(admitted=True,index=len(a['valid'])-1,feature_receipt=ar)
  details.append(meta)
 pred=p.model.predict(a,None)
 for d in details:
  if d['admitted']:
   i=d['index'];d.update(NC=float(pred['raw_physical_nc'][i]),DAC=float(pred['raw_dac_event'][i]),eligible=bool(pred['eligible'][i]),score=float(pred['final_scores'][i]))
 receipt['timing_probe']=dict(attempts=details,choice=int(pred['choice']),old_choice=int(saved['choice']));return a,asset,native,current,receipt
p.base_inputs=inputs;p.plan(history);rows=list(map(json.loads,open(O/'policy/plans.jsonl')));r=rows[-1]['base_receipt']['timing_probe'];(O/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n');(O/'history.json').write_text(json.dumps(history)+'\n');print(json.dumps(r),flush=True)
