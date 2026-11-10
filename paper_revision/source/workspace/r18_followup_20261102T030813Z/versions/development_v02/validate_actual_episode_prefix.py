"""Compare each saved controller prefix with the actual following 20Hz motion."""
from pathlib import Path
import json,math
import numpy as np
def validate(trace):
    trace=Path(trace);records=[json.loads(s)for s in (trace/'trace.jsonl').read_text().splitlines()];plans=[json.loads(s)for s in (trace/'policy/plans.jsonl').read_text().splitlines()];times=np.asarray([x['sim_seconds']for x in records]);starts={r['selected_replan']:i for i,r in enumerate(records)if r['selected_replan']is not None};errors=[];headings=[];first=[];rows=[];complete=0
    for p in plans:
        n=p['replan'];i=starts[n];r=records[i];detail=p['base_receipt']['supply_v2']['physical_checks'][p['choice']]['prefix_model'];commands=detail['commands'];c=r['control'];first_error=float(abs(np.array([c['steer'],c['throttle'],c['brake']])-commands[0]).max());first.append(first_error)
        with np.load(trace/'policy'/f'plan_{n:05d}.npz')as a:path=a['controller_prefix_local'][p['choice']]
        yaw=r['body_pose'][2];R=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]]);origin=np.array(r['actual_rear']);e=[];h=[]
        for j in range(1,11):
            ti=r['sim_seconds']+j*.05;k=int(np.argmin(abs(times-ti)))
            if abs(times[k]-ti)>1e-4:continue
            actual=(np.array(records[k]['actual_rear'])-origin)@R;angle=records[k]['body_pose'][2]-yaw;angle=math.atan2(math.sin(angle-path[j,2]),math.cos(angle-path[j,2]));e.append(float(np.linalg.norm(actual-path[j,:2])));h.append(abs(angle))
        if len(e)==10:complete+=1
        errors.extend(e);headings.extend(h);rows.append(dict(replan=n,first_control_abs_error=first_error,matched_samples=len(e),max_position_error_m=max(e,default=None),max_heading_error_rad=max(h,default=None)))
    p95=float(np.percentile(errors,95))if errors else None;h95=float(np.percentile(headings,95))if headings else None;firstmax=max(first,default=None)
    return dict(passed=bool(complete and firstmax<=1e-5 and p95<=.50 and h95<=.20),complete_half_second_prefixes=complete,matched_motion_samples=len(errors),first_control_max_abs_error=firstmax,position_error_p95_m=p95,position_error_max_m=max(errors,default=None),heading_error_p95_rad=h95,thresholds=dict(first_control_abs_error=1e-5,position_error_p95_m=.50,heading_error_p95_rad=.20),scope='Observed own-policy actual motion; model discrepancy diagnostic, not absolute safety or collision-free guarantee',rows=rows)
