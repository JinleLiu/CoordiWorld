"""Read-only prefix diagnostic; no live policy, scorer, or model changes."""
import argparse, copy, csv, hashlib, importlib.util, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from world4scorer_agent import World4ScorerAgent, PID

ROOT=Path('<CW_WORKSPACE>/twentyeighth_20261031')
GARAGE=ROOT/'external/carla_garage-48e6a6dbcb133220333bd65e242fbc2b492bb15b/team_code'
WRAP=lambda x:(x+np.pi)%(2*np.pi)-np.pi
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Hero:
 def __init__(self):self.speed=0.
 def get_velocity(self):return SimpleNamespace(x=self.speed,y=0.)
def agent():
 a=World4ScorerAgent.__new__(World4ScorerAgent);a.hero_actor=Hero();a.turn_pid=PID(1.25,.75,.3);a.speed_pid=PID(5.,.5,1.);a.still_ticks=0;a.creep_left=0;a.creep_cool=0;a.plan=None;a.plan_t0=0.;return a
def ctrl(a,t,pose,v):
 a.hero_actor.speed=v;c=a._track(t,np.asarray(pose));return np.array([c.steer,c.throttle,c.brake])
def rear(p,off):
 c,s=np.cos(p[2]),np.sin(p[2]);return p[:2]+np.array([[c,-s],[s,c]])@off
def garage_step(model,p,v,u):
 loc,ang,vel=model.forecast_ego_vehicle(np.array([p[0],-p[1],0.]),np.array([-p[2]]),np.array([v]),u)
 return np.array([loc[0],-loc[1],-float(np.asarray(ang).ravel()[0])]),float(np.asarray(vel).ravel()[0])
def run(args):
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 reg=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),scope=f'Already exposed Dev10, World4Scorer_EC seed{args.seed}; all completed 0.5s intervals before next replan; development diagnostic, no training',hypothesis='The published native PID plus CARLA-tuned bicycle can reduce current Pacifica execution-prefix error',inputs='Current own pose/speed, saved selected plan, native PID and creep state reconstructed from preceding observations only',label='Observed own pose at t+0.5s; never used in predictions',comparisons=['saved NAVSIM Pacifica LQR','constant current world velocity and heading','native W4 PID + published carla_garage bicycle','published bicycle with observed future controls (NONCAUSAL dynamics diagnostic only)'],primary='mean per-route mean rear-position error on initial horizontal speed >0.5 m/s; pooled p95 also reported',continue_condition='At least 20% primary reduction against Pacifica and no worsening pooled moving p95; this is only an execution-proxy research signal, not policy adoption',native_control_replay_tolerance=1e-6,model_selection=False,policy_changes=False,source_commit='48e6a6dbcb133220333bd65e242fbc2b492bb15b',time_step=.05,parameters='Unchanged official GlobalConfig; no fitting')
 (out/'REGISTRATION.json').write_text(json.dumps(reg,indent=2)+'\n')
 config=load('config',GARAGE/'config.py').GlobalConfig();model=load('garage_motion_pinned',GARAGE/'kinematic_bicycle_model.py').KinematicBicycleModel(config)
 rows=[];checks=[];sources=[];start=time.perf_counter()
 for d in sorted((ROOT/f'bench2drive/episodes/seed{args.seed}').glob('*/World4Scorer_EC/attempt1/trace'),key=lambda p:int(p.parts[-4])):
  route=d.parts[-4]
  if args.route and route!=args.route:continue
  trace=[json.loads(x)for x in open(d/'trace.jsonl')];states=[json.loads(x)for x in open(d/'transfer_trace.jsonl')];plans=[json.loads(x)for x in open(d/'transfer_plans.jsonl')];meta=json.load(open(d/'agent_metadata.json'));off=np.asarray(meta['geometry']['rear_offset_math']);assert len(trace)==len(states)
  a=agent();last=-1;errors=[];n_before=len(rows)
  for i,t in enumerate(trace):
   v=float(np.linalg.norm(t['velocity_world'][:2]));pose=np.asarray(t['pose']);ri=t['replans']-1;is_new=ri!=last
   if is_new:
    ar=np.load(d/f'transfer_{ri:05d}.npz');ch=int(ar['choice']);candidate=ar['candidates_body'][ch];c,s=np.cos(pose[2]),np.sin(pose[2]);rot=np.array([[c,-s],[s,c]]);a.plan=np.zeros((9,3));a.plan[0]=pose;a.plan[1:,:2]=pose[:2]+candidate[:,:2]@rot.T;a.plan[1:,2]=pose[2]+candidate[:,2];a.plan_t0=t['sim_seconds'];last=ri
   before=copy.deepcopy(a)if is_new else None
   predicted=ctrl(a,t['sim_seconds'],pose,v);observed=np.array([t['control'][k]for k in ['steer','throttle','brake']]);err=float(np.max(abs(predicted-observed)));errors.append(err)
   if not is_new or i+10>=len(trace):continue
   target=trace[i+10];dt=target['sim_seconds']-t['sim_seconds']
   if abs(dt-.5)>1e-5 or any(trace[j]['replans']!=t['replans']for j in range(i,i+10)):continue
   actual_pose=np.asarray(target['pose']);actual_rear=np.asarray(states[i+10]['state']['rear_pose'][:2]);np.testing.assert_allclose(rear(actual_pose,off),actual_rear,rtol=0,atol=1e-8)
   pac=np.array(ar['native'][ch,5,:3]);cv=pose.copy();cv[:2]+=np.asarray(t['velocity_world'][:2])*dt
   pg=pose.copy();vg=v;po=pose.copy();vo=v
   for k in range(10):
    u=ctrl(before,t['sim_seconds']+k*.05,pg,vg);pg,vg=garage_step(model,pg,vg,u)
    observed_k=np.array([trace[i+k]['control'][z]for z in ['steer','throttle','brake']]);po,vo=garage_step(model,po,vo,observed_k)
   row=dict(route=route,seed=args.seed,replan=ri,sim_s=t['sim_seconds'],dt=dt,initial_speed_mps=v,moving=int(v>.5),choice=ch,control_replay_error=err)
   for name,p in [('pacifica',pac),('constant_velocity',cv),('garage_pid',pg),('garage_observed_control_NONCAUSAL',po)]:
    xy=p[:2]if name=='pacifica'else rear(p,off);row[name+'_position_error_m']=float(np.linalg.norm(xy-actual_rear));row[name+'_heading_error_rad']=float(abs(WRAP(p[2]-actual_pose[2])))
   rows.append(row)
  check=dict(route=route,ticks=len(trace),prefixes=len(rows)-n_before,max_native_control_replay_error=max(errors),nonmatching_ticks=sum(x>1e-6 for x in errors));checks.append(check);print(json.dumps(check),flush=True)
  for p in [d/'trace.jsonl',d/'transfer_trace.jsonl',d/'transfer_plans.jsonl',d/'agent_metadata.json']:sources.append(dict(path=str(p),sha256=digest(p),size=p.stat().st_size))
  if max(errors)>1e-6:raise AssertionError(('native control reconstruction failed',check))
 assert rows
 with open(out/'per_prefix.csv','w',newline='')as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 methods=['pacifica','constant_velocity','garage_pid','garage_observed_control_NONCAUSAL'];summary={};drive_rows=[]
 for scope,rr in [('all',rows),('moving_initial_speed_gt0p5',[r for r in rows if r['moving']]),('stationary',[r for r in rows if not r['moving']])]:
  summary[scope]=dict(n=len(rr),routes=len(set(r['route']for r in rr)),methods={})
  for m in methods:
   vals=[r[m+'_position_error_m']for r in rr];route_means=[]
   for route in sorted(set(r['route']for r in rr)):
    rv=[r[m+'_position_error_m']for r in rr if r['route']==route];route_means.append(float(np.mean(rv)));drive_rows.append(dict(scope=scope,route=route,method=m,n=len(rv),mean_position_error_m=float(np.mean(rv)),p95_position_error_m=float(np.quantile(rv,.95))))
   summary[scope]['methods'][m]=dict(pooled_mean_position_error_m=float(np.mean(vals))if vals else None,pooled_p95_position_error_m=float(np.quantile(vals,.95))if vals else None,route_equal_mean_position_error_m=float(np.mean(route_means))if vals else None,pooled_mean_heading_error_rad=float(np.mean([r[m+'_heading_error_rad']for r in rr]))if rr else None)
 with open(out/'per_route.csv','w',newline='')as f:w=csv.DictWriter(f,list(drive_rows[0]));w.writeheader();w.writerows(drive_rows)
 moving=summary['moving_initial_speed_gt0p5']['methods'];baseline=moving['pacifica'];new=moving['garage_pid'];signal=new['route_equal_mean_position_error_m']<=.8*baseline['route_equal_mean_position_error_m']and new['pooled_p95_position_error_m']<=baseline['pooled_p95_position_error_m']
 source_files=[GARAGE/'config.py',GARAGE/'kinematic_bicycle_model.py',Path(sys.modules['world4scorer_agent'].__file__),Path(__file__)]
 for p in source_files:sources.append(dict(path=str(p),sha256=digest(p),size=p.stat().st_size))
 result=dict(registration=reg,summary=summary,control_replay=checks,seconds=time.perf_counter()-start,signal=signal,decision='Execution-prefix signal only; no policy change or 4s/EC validity established'if signal else'No continuation signal at registered threshold; retain as negative execution diagnostic',sources=sources)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(summary=summary,signal=signal,seconds=result['seconds'])),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'research/garage_execution_diagnostic'));p.add_argument('--route');p.add_argument('--seed',type=int,default=17);run(p.parse_args())
