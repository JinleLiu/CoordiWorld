"""Saved-state repair audit, without simulation, new outcomes or new labels.

Only signed-negative initial velocity can enter the changed integrator branch.
For v>=0, vnew=max(0,v+acc*dt) inductively preserves the original computation.
Unchanged non-CV rejection also proves admission stays false. All other
candidate CV admissions are recomputed, including attempted/rejected supply.
Existing FINALs and their original failed gate are never overwritten.
"""
from pathlib import Path
import json,sys,datetime,hashlib,math,os,fcntl,traceback,importlib.util
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
R=Path('<CW_WORKSPACE>/r18_followup_20261102T030813Z');OUT=R/'diagnostics/interface_repair_v02'
sys.path.insert(0,str(R/'versions/interface_repair_v02'))
from cw_controller_prefix import roll
from cw_followup_physics import cv_box_check
spec=importlib.util.spec_from_file_location('frozen_old_prefix',R/'versions/development_v01/cw_controller_prefix.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def readlines(p):return [json.loads(s)for s in Path(p).read_text().splitlines()]
def check_cv(fn,f,ca,ex,physics):
 path,d=fn(f,ca,ex,f['controller_state'],physics);cv=cv_box_check(path,f,dt=.05,stationary_speed=.2)
 if f['controller_state']['actual_control']['gear']==0:
  engaged,_=fn(f,ca,ex,f['controller_state'],physics,assume_engaged=True)
  cv['cv_clear']=bool(cv['cv_clear']and all(cv_box_check(path+w*(engaged-path),f,dt=.05,stationary_speed=.2)['cv_clear']for w in [.25,.5,.75,1.]))
 return path,d,cv
def audit(eptext):
 ep=Path(eptext);tr=ep/'attempt1/trace';plans=readlines(tr/'policy/plans.jsonl');frames=readlines(tr/'observed_history.jsonl');fi={x['timestamp']:i for i,x in enumerate(frames)};traces=readlines(tr/'trace.jsonl');starts={x['selected_replan']:i for i,x in enumerate(traces)if x['selected_replan']is not None};physics=json.loads((tr/'agent_metadata.json').read_text())['supply_physics']
 counts=dict(plans=0,negative_initial_velocity_plans=0,pool_entries=0,supply_attempts=0,forward_invariant_prefixes=0,nonCV_invariant_rejections=0,recomputed_pool_prefixes=0,recomputed_supply_prefixes=0,regenerated_rejected_native=0,skipped_original_geometry_invalid=0)
 changes=[];allerrors=[];headings=[];first=[];complete=0;details=[]
 for p in plans:
  n=p['replan'];fidx=fi[p['timestamp_us']];f=frames[fidx];h=frames[fidx-3:fidx+1];negative=float(f['ego_dynamic_state'][0])<0.;counts['plans']+=1;counts['negative_initial_velocity_plans']+=int(negative);s=p['base_receipt']['supply_v2'];i0=starts[n];r=traces[i0];selected=p['choice'];recomputed={}
  with np.load(tr/'policy'/f'plan_{n:05d}.npz',allow_pickle=False)as a:
   for i,rr in enumerate(s['physical_checks']):
    counts['pool_entries']+=1
    if 'prefix_model'not in rr:counts['skipped_original_geometry_invalid']+=1;continue
    assert bool(a['common_physical_mask'][i])==rr['admitted_physics']
    if not negative:counts['forward_invariant_prefixes']+=1;continue
    other=[v for v in rr['reject_reasons']if v!='controller_prefix_dynamic_bbox']
    if other and i!=selected:counts['nonCV_invariant_rejections']+=1;continue
    path,d,cv=check_cv(roll,f,a['candidates'][i],a['native'][i],physics);counts['recomputed_pool_prefixes']+=1;recomputed[i]=(path,d)
    good=bool(cv['cv_clear']and not other)
    if good!=rr['admitted_physics']:changes.append(dict(replan=n,kind='pool',index=i,old=rr['admitted_physics'],new=good,old_cv=rr['controller_prefix_cv'],new_cv=cv))
   for attempt in s['attempts']:
    counts['supply_attempts']+=1
    if 'execution_check'not in attempt:continue # unchanged geometric/duplicate rejection
    rr=attempt['execution_check']
    if not negative:counts['forward_invariant_prefixes']+=1;continue
    other=[v for v in rr['reject_reasons']if v!='controller_prefix_dynamic_bbox']
    if other:counts['nonCV_invariant_rejections']+=1;continue
    if attempt['admitted']:
     i=attempt['candidate_index'];assert i in recomputed
     # Same candidate was already covered in the pool audit.
     continue
    from revision_v5_twentyfourth_prepare import predict_vector_controller_execution
    ca=np.asarray(attempt['candidate'],dtype=np.float32);_,fn,_=predict_vector_controller_execution(h,ca[None],np.array([True]));ex=fn[0];counts['regenerated_rejected_native']+=1
    op,od,ocv=check_cv(old.roll,f,ca,ex,physics)
    np.testing.assert_allclose(od['commands'],rr['prefix_model']['commands'],atol=1e-7,rtol=0)
    assert ocv['cv_clear']==rr['controller_prefix_cv']['cv_clear']
    assert abs(ocv['min_separation']-rr['controller_prefix_cv']['min_separation'])<1e-6
    path,d,cv=check_cv(roll,f,ca,ex,physics);counts['recomputed_supply_prefixes']+=1
    good=bool(cv['cv_clear']and not other)
    if good!=rr['admitted_physics']:changes.append(dict(replan=n,kind='attempt',attempt=attempt['attempt'],old=rr['admitted_physics'],new=good,old_cv=rr['controller_prefix_cv'],new_cv=cv))
   if selected in recomputed:path,model=recomputed[selected]
   else:path=a['controller_prefix_local'][selected];model=s['physical_checks'][selected]['prefix_model']
   actual=np.array([r['control'][k]for k in ['steer','throttle','brake']]);err=float(abs(actual-np.asarray(model['commands'][0])).max());first.append(err)
   yaw=r['body_pose'][2];rot=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]]);origin=np.asarray(r['actual_rear']);ee=[];hh=[]
   for j in range(1,11):
    k=i0+j
    if k>=len(traces)or abs(traces[k]['sim_seconds']-r['sim_seconds']-j*.05)>1e-4:continue
    actual=(np.array(traces[k]['actual_rear'])-origin)@rot;angle=traces[k]['body_pose'][2]-yaw-path[j,2];ee.append(float(np.linalg.norm(actual-path[j,:2])));hh.append(abs(math.atan2(math.sin(angle),math.cos(angle))))
   complete+=len(ee)==10;allerrors.extend(ee);headings.extend(hh);details.append(dict(replan=n,signed_vx=float(f['ego_dynamic_state'][0]),negative_branch_recomputed=negative,first_control_abs_error=err,matched_samples=len(ee)))
  if n%50==0:write(OUT/(ep.parent.name+'__'+ep.name+'.progress.json'),dict(pid=os.getpid(),counts=counts,changes=len(changes)))
 p95=float(np.percentile(allerrors,95));h95=float(np.percentile(headings,95));fm=max(first);passed=bool(complete and fm<=1e-5 and p95<=.5 and h95<=.2)
 result=dict(episode=str(ep),original_final_sha256=sha(ep/'FINAL.json'),counts=counts,admission_changes=changes,admission_equivalent=not changes,corrected_selected_prefix_gate=dict(passed=passed,complete_half_second_prefixes=complete,matched_motion_samples=len(allerrors),first_control_max_abs_error=fm,position_error_p95_m=p95,position_error_max_m=max(allerrors),heading_error_p95_rad=h95,thresholds=dict(first_control_abs_error=1e-5,position_error_p95_m=.5,heading_error_p95_rad=.2)),reuse_eligible=passed and not changes,source_prefix_sha256=sha(R/'versions/interface_repair_v02/cw_controller_prefix.py'),rows=details)
 write(OUT/(ep.parent.name+'__'+ep.name+'.json'),result);return {k:v for k,v in result.items()if k!='rows'}
def main():
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=datetime.datetime.now(datetime.timezone.utc).isoformat();eps=[str(p.parent)for p in sorted((R/'development_v01/episodes').glob('seed*/*/*/FINAL.json'))];assert len(eps)==8
 write(OUT/'START.json',dict(utc=start,pid=os.getpid(),birth_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],scope='Read-only saved-state audit, no simulator/episode',episodes=eps,workers=3,source_sha256=sha(__file__)))
 results=[];fail=[]
 with ProcessPoolExecutor(max_workers=3)as pool:
  jobs={pool.submit(audit,ep):ep for ep in eps}
  for fut in as_completed(jobs):
   try:results.append(fut.result())
   except Exception:fail.append(dict(episode=jobs[fut],error=traceback.format_exc()))
   write(OUT/'STATUS.json',dict(completed=len(results),errors=fail,expected=8))
 result=dict(started_utc=start,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results=results,errors=fail,passed=len(results)==8 and not fail and all(r['reuse_eligible']for r in results),new_episodes=0,source_sha256=sha(__file__),scope='Full logged pool/attempt admission accounting; signed-negative frames recomputed and nonnegative integrator is structurally unchanged by induction; unrelated static/geometric rejection preserved. Corrected selected actual-motion gate recomputed with unchanged thresholds. Original FINALs retain original failed gates.')
 write(OUT/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items()if k!='results'}),flush=True)
if __name__=='__main__':main()
