"""Current-input topology/coverage of four registered progress failures."""
import json,hashlib
from pathlib import Path
import numpy as np
from revision_v5_twentyeighth_map import CarlaMap
from revision_v5_twentyeighth_map_supply import MapSupply,coverage,separation,local_execution
from revision_v5_twentyeighth_policy import array_history
from revision_v5_twentyeighth_w4_execution import Executor
B=Path('<CW_WORKSPACE>/twentyeighth_20261031');O=B/'map_supply/topology_cases';O.mkdir(parents=True,exist_ok=True);maps={};res=[];executor=Executor()
for route in ['3514','25424','27494','2091']:
 d=B/f'bench2drive/episodes/seed17/{route}/CoordiWorld_BaseEC/attempt1/trace';plans=list(map(json.loads,open(d/'policy/plans.jsonl')));history=list(map(json.loads,open(d/'observed_history.jsonl')));byts={h['timestamp']:h for h in history};town=json.load(open(d/'agent_metadata.json'))['map_name'].split('/')[-1]
 if town not in maps:maps[town]=MapSupply(CarlaMap(json.load(open(B/f'bridge/maps/{town}.json'))))
 fac=maps[town];initial=np.asarray(byts[plans[0]['timestamp_us']]['ego2global_translation'][:2]);moved=False;stall=None
 for row in plans:
  h=byts[row['timestamp_us']];xy=np.asarray(h['ego2global_translation'][:2]);moved|=np.linalg.norm(xy-initial)>1.;prev=[z for z in history if z['timestamp']<=h['timestamp']-1000000]
  if moved and prev and np.linalg.norm(h['ego_dynamic_state'][:2])<.5 and np.linalg.norm(xy-np.asarray(prev[-1]['ego2global_translation'][:2]))<.25:stall=row['replan'];break
 for i in dict.fromkeys([0,stall,len(plans)-1]):
  if i is None:continue
  row=plans[i];h=array_history([byts[row['timestamp_us']]])[0];file=d/'policy'/f'plan_{i:05d}.npz';a=np.load(file);lanes,top=fac.targets(h);cov,sat=coverage(a,a['native'],dict(eligible=a['PNC_eligible']),h,lanes);props,top=fac.generate(h);new=[]
  for ca,detail,_ in props:
   detail=dict(detail)
   if ca is not None:
    st=a['native'][np.flatnonzero(a['valid'])[0],0];init=executor.initial(dict(timestamp_us=h['timestamp'],rear_pose=st[:3].tolist(),rear_velocity_body=st[3:5].tolist(),rear_acceleration_body=st[5:7].tolist(),tire_angle_surrogate=float(st[7]),yaw_rate=float(st[9]),yaw_acceleration=float(st[10])));native=executor.execute(ca[None],init)[0];margin,actor=separation(local_execution(native,h),h);detail.update(executed_minSAT=margin,executed_actor=actor,executed_forward=float(local_execution(native,h)[-1,0]))
   new.append(detail)
  rec=dict(route=route,source=str(file),source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),replan=i,sim_seconds=h['sim_seconds'],case='initial'if i==0 else'first_post_motion_stall'if i==stall else'last',topology=top,safe_forward_indices=np.flatnonzero(cov).tolist(),selected=int(a['choice']),eligible=np.flatnonzero(a['PNC_eligible']).tolist(),valid=np.flatnonzero(a['valid']).tolist(),current_stationary_SAT_m=sat.tolist(),attempts=new,interpretation='Current map/static-box diagnostic, no counterfactual real driving outcome or learned risk recomputation')
  res.append(rec);print(json.dumps(rec),flush=True)
(O/'RESULT.json').write_text(json.dumps(res,indent=2)+'\n')
