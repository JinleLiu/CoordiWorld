"""Actual CARLA actuation from frozen CoordiWorld selected trajectories.

PID coefficients/lookahead follow official W4S agent at 7814310e; creep is
not part of this shared CW tracker. Full/BaseEC have identical adapters.
"""
import json,time,math,hashlib
from collections import deque
from pathlib import Path
import numpy as np,carla
from leaderboard.autoagents.autonomous_agent import AutonomousAgent,Track
from revision_v5_twentyeighth_carla_view import pose,rot,frame,export_map,vehicle_geometry
from revision_v5_twentyeighth_policy import BridgePolicy
B=Path('<CW_WORKSPACE>/twentyeighth_20261031')
def get_entry_point():return 'CoordiWorldAgent'
class PID:
 def __init__(self,kp,ki,kd):self.kp=kp;self.ki=ki;self.kd=kd;self.win=deque([0.]*40,maxlen=40)
 def step(self,e):self.win.append(e);return self.kp*e+self.ki*np.mean(self.win)+self.kd*(self.win[-1]-self.win[-2])
class CoordiWorldAgent(AutonomousAgent):
 def set_global_plan(self,gps,world):self.dense=list(world);super().set_global_plan(gps,world)
 def setup(self,path):
  t=time.perf_counter();self.track=Track.MAP;self.cfg=json.load(open(path.split('+')[0]));self.out=Path(self.cfg['output_dir']);self.out.mkdir(parents=True,exist_ok=True);self.geometry=vehicle_geometry(self.hero_actor)
  cm=self.hero_actor.get_world().get_map();self.mapdata=export_map(cm,B/'bridge/maps'/f"{cm.name.split('/')[-1]}.json");self.policy=BridgePolicy(self.mapdata,self.cfg['variant'],self.out/'policy',self.cfg.get('nominal_mode','planned'))
  self.history=[];self.tick=0;self.plan=None;self.native=None;self.plan_t0=0.;self.turn=PID(1.25,.75,.3);self.speedpid=PID(5.,.5,1.);self.trace=open(self.out/'trace.jsonl','w',buffering=1);self.observed=open(self.out/'observed_history.jsonl','w',buffering=1);self.episode=hashlib.sha256(path.encode()).hexdigest()[:16]
  meta=dict(method='CoordiWorld_'+self.cfg['variant'],setup_seconds=time.perf_counter()-t,geometry=self.geometry,map_name=cm.name,map_sha=self.mapdata['opendrive_sha'],config=self.cfg,permissions='actual current structured actors within100m, map and navigation route, own past observations/plans; no true future actors',history_start='brake until four actual observations at0.5s spacing',replan_hz=2,control_hz=20,controller='same PID forFull/BaseEC; W4S gains/lookahead; no creep or BasicAgent decisions',execution_model='frozen native NAVSIM Pacifica surrogate, distinct from CARLA actuation')
  (self.out/'agent_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 def sensors(self):return [dict(type='sensor.speedometer',reading_frequency=20,id='speed'),dict(type='sensor.other.imu',x=0.,y=0.,z=0.,roll=0.,pitch=0.,yaw=0.,sensor_tick=.05,id='imu')]
 def at(self,tau):
  tau=np.clip(tau,0,3.99);i=min(int(tau/.5),7);f=(tau-i*.5)/.5;return self.plan[i,:2]*(1-f)+self.plan[i+1,:2]*f
 def tracker(self,timestamp,body):
  ctrl=carla.VehicleControl()
  if self.plan is None:ctrl.brake=1.;return ctrl
  tau=timestamp-self.plan_t0;v=self.hero_actor.get_velocity();speed=math.hypot(v.x,v.y);dp=self.at(tau+.5)-self.at(tau);c,s=math.cos(body[2]),math.sin(body[2]);desired=max(0.,c*dp[0]+s*dp[1])*2.
  if self.cfg.get('speed_source')=='native_preview_v02':
   # The selected smooth execution starts at measured v/a. A half-second
   # mean displacement below the inherited brake floor can latch at rest.
   # Preview its predicted speed; no minimum throttle or creep override.
   desired=max(0.,float(np.interp(min(tau+self.cfg['speed_preview_seconds'],4.),np.arange(41)*.1,self.native[:,3])))
  dist=10. if speed>6.5 else 4.;aim=self.at(3.99)
  for tt in np.arange(max(0.,tau),4.,.05):
   a=self.at(tt)
   if np.linalg.norm(a-body[:2])>=dist:aim=a;break
  d=aim-body[:2];fwd=c*d[0]+s*d[1];lat=-s*d[0]+c*d[1];angle=0. if fwd<.5 else math.degrees(math.atan2(lat,fwd))/90.;brake=desired<.15 or speed/max(desired,.1)>1.1
  ctrl.steer=float(np.clip(-self.turn.step(angle),-1,1));thr=float(np.clip(self.speedpid.step(float(np.clip(desired-speed,0,.25))),0,.75));ctrl.throttle=0. if brake else thr;ctrl.brake=float(brake);self.desired=desired;return ctrl
 def run_step(self,data,timestamp):
  begin=time.perf_counter();body=pose(self.hero_actor.get_transform());receipt=None;replan_seconds=0.
  if self.tick%10==0:
   f=frame(self.hero_actor,float(timestamp),self.geometry,self.mapdata,self.dense,self.episode,self.history[-1]if self.history else None);self.history.append(f);self.history=self.history[-4:];self.observed.write(json.dumps(f)+'\n')
   if len(self.history)==4:
    if self.policy.n==0:(self.out/'first_four_frames.json').write_text(json.dumps(self.history))
    t=time.perf_counter();ca,native,receipt=self.policy.plan(self.history);replan_seconds=time.perf_counter()-t;rear=np.array(f['ego2global_translation'][:2]);yaw=body[2];ra=np.r_[np.r_[rear,yaw][None],np.c_[ca[:,:2]@rot(yaw).T+rear,np.arctan2(np.sin(ca[:,2]+yaw),np.cos(ca[:,2]+yaw))]]
    off=np.asarray(self.geometry['rear_offset_math']);self.plan=ra.copy()
    for k in range(len(ra)):self.plan[k,:2]-=rot(ra[k,2])@off
    self.native=native;self.plan_t0=float(timestamp)
  ctrl=self.tracker(float(timestamp),body);v=self.hero_actor.get_velocity();actual_rear=body[:2]+rot(body[2])@np.asarray(self.geometry['rear_offset_math']);err=None
  if self.native is not None:
   tau=timestamp-self.plan_t0;pred=np.array([np.interp(tau,np.arange(41)*.1,self.native[:,i])for i in [0,1]]);err=float(np.linalg.norm(pred-actual_rear))
  self.trace.write(json.dumps(dict(step=self.tick,sim_seconds=float(timestamp),body_pose=body.tolist(),actual_rear=actual_rear.tolist(),velocity_world=[v.x,-v.y,v.z],control=dict(steer=ctrl.steer,throttle=ctrl.throttle,brake=ctrl.brake),selected_replan=None if receipt is None else receipt['replan'],choice=None if receipt is None else receipt['choice'],replan_seconds=replan_seconds,run_step_seconds=time.perf_counter()-begin,native_tracking_error_m=err,desired_speed=getattr(self,'desired',None),policy_replans=self.policy.n))+'\n');self.tick+=1;return ctrl
 def destroy(self):
  if hasattr(self,'trace'):self.trace.close();self.observed.close()
