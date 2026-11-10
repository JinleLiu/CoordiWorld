"""Binding/first-command checks; no shadow score is a measured outcome."""
from pathlib import Path
import json,hashlib,math
import numpy as np
def digest(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def validate(trace):
 tr=Path(trace);plans=[json.loads(s)for s in(tr/'policy/plans.jsonl').read_text().splitlines()];prefix=[json.loads(s)for s in(tr/'policy/v05_prefix.jsonl').read_text().splitlines()];actual=[json.loads(s)for s in(tr/'trace.jsonl').read_text().splitlines()]
 assert len(plans)==len(prefix)>0
 starts={r['selected_replan']:i for i,r in enumerate(actual)if r.get('selected_replan')is not None};times=np.array([r['sim_seconds']for r in actual]);rows=[];errors=[];angles=[]
 for p,q in zip(plans,prefix):
  n=p['replan'];assert q['replan']==n and q['choice']==p['choice'] and q['source_sha']==p['source_sha'];assert q['candidate_sha']==p['chosen_candidate_sha'] and q['native_sha']==p['chosen_execution_sha']
  assert not p.get('map_error') and not p.get('pdm_error'),('input_error',n,p.get('map_error'),p.get('pdm_error'))
  with np.load(tr/'policy'/f'plan_{n:05d}.npz')as a:
   ch=int(a['choice']);assert ch==p['choice'] and a['valid'][ch]
   assert a['PNC_eligible'][ch],('selected_outside_final_eligibility',n)
   assert digest(a['candidates'][ch])==q['candidate_sha'] and digest(a['native'][ch])==q['native_sha']
   assert not p['base_receipt'].get('supply_v2'),'Supply-v2 leaked into historical platform'
  with np.load(tr/'policy'/f'v05_selected_prefix_{n:05d}.npz')as a:
   predicted=a['prefix'];commands=a['commands'];assert digest(a['selected_native'])==q['native_sha']
  idx=starts[n];r=actual[idx];ctrl=r['control'];err=float(np.max(np.abs(commands[0]-[ctrl['steer'],ctrl['throttle'],ctrl['brake']])))
  assert err<=1e-5,('first_control_binding',n,err)
  yaw=r['body_pose'][2];R=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]]);origin=np.array(r['actual_rear']);e=[];h=[]
  for j in range(1,11):
   k=int(np.argmin(abs(times-(r['sim_seconds']+j*.05))))
   if abs(times[k]-(r['sim_seconds']+j*.05))>1e-4:continue
   obs=(np.array(actual[k]['actual_rear'])-origin)@R;ang=actual[k]['body_pose'][2]-yaw-predicted[j,2];e.append(float(np.linalg.norm(obs-predicted[j,:2])));h.append(abs(math.atan2(math.sin(ang),math.cos(ang))))
  errors+=e;angles+=h;rows.append(dict(replan=n,first_control_error=err,motion_samples=len(e),max_position_error_m=max(e,default=None)))
 return dict(passed=True,plans=len(plans),binding_checks='candidate/native/head eligibility/source/timestamp/first actual command exact',Supply_v2_gate_imported=False,prefix_motion_samples=len(errors),prefix_position_p95_m=float(np.percentile(errors,95))if errors else None,prefix_heading_p95_rad=float(np.percentile(angles,95))if angles else None,prefix_discrepancy_is_diagnostic_not_new_policy_gate=True,rows=rows)
if __name__=='__main__':
 import sys
 p=Path(sys.argv[1]);out=validate(p);(p.parent/'V05_INTERFACE_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='rows'}))
