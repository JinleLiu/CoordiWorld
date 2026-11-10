"""Frozen W4S + EC / EC-H3 / EC-Gspeed, native cameras/route/PID/creep intact."""
import json,time,math,subprocess,hashlib
from pathlib import Path
import numpy as np,torch
from world4scorer_agent import model_config,ROUTE_ALIGN,B2D_ORDER
from revision_v5_twentyeighth_w4s_agent import R18World4ScorerAgent
from revision_v5_twentyeighth_w4_rescore import FrozenPoolScorer
from revision_v5_twentyeighth_w4_execution import read_packet,send_packet,wrap
from revision_v5_twentyeighth_carla_view import vehicle_geometry,rot,world_vector
B=Path('<CW_WORKSPACE>/twentyeighth_20261031')
def get_entry_point():return 'R18W4TransferAgent'
class R18W4TransferAgent(R18World4ScorerAgent):
 def setup(self,path):
  super().setup(path);self.variant=self.r18cfg['variant'];assert self.variant in ['EC','EC_H3','EC_Gspeed']and self.r18cfg['ec_lambda']==1.
  self.pool_scorer=FrozenPoolScorer(self.model);self.geometry=vehicle_geometry(self.hero_actor);self.previous_observation=None;self.previous_native=None;self.previous_timestamp_us=None;self.last_transfer=None
  self.worker_log=open(self.out/'execution_worker.log','w');self.worker=subprocess.Popen([str(B/'scripts/cw_bridge_python.sh'),str(B/'scripts/revision_v5_twentyeighth_w4_execution.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.worker_log)
  self.transfer_trace=open(self.out/'transfer_trace.jsonl','w',buffering=1)
  p=self.out/'agent_metadata.json';m=json.load(open(p));m.update(method=self.r18cfg['method_id'],identity='official-pretrained-frozen-plus-registered-selection',wrapper='frozen native perception and PID; registered EC/supply selection; no new weights',route_world=[[t.location.x,t.location.y,t.location.z,t.rotation.roll,t.rotation.pitch,t.rotation.yaw,str(o)]for t,o in self._dense_plan_world],variant=self.variant,ec_lambda=1.,P0=False,execution_model='pinned NAVSIM Pacifica LQR/bicycle in isolated process; CARLA control remains official W4S PID/creep',history='own selected predicted execution only; reset each episode',geometry=self.geometry,additional_ego_terms='yaw rate/acceleration derived from own observed pose history; no surrounding actors or future observations',generation='trigger when all original64 executed EC are0; H3 and Gspeed at most3 attempts, same duplicate/EC admission',scoring='entire augmented pool reruns frozen candidate self-attention and score heads; route bonus uses original model score spread of this pool, before EC penalty');p.write_text(json.dumps(m,indent=2)+'\n')
 def observe(self,timestamp):
  body=self._pose();omega=alpha=0.;known=self.previous_observation is not None;alpha_known=False
  if known:
   dt=timestamp-self.previous_observation['time'];assert dt>0;omega=float(wrap(body[2]-self.previous_observation['heading'])/dt);alpha_known=self.previous_observation['omega_known'];alpha=(omega-self.previous_observation['omega'])/dt if alpha_known else 0.
  off=np.asarray(self.geometry['rear_offset_math']);R=rot(body[2]);r=R@off;vg=world_vector(self.hero_actor.get_velocity());ag=world_vector(self.hero_actor.get_acceleration());vr=R.T@(vg+omega*np.array([-r[1],r[0]]));ar=R.T@(ag+alpha*np.array([-r[1],r[0]])-omega*omega*r)
  self.previous_observation=dict(time=float(timestamp),heading=float(body[2]),omega=omega,omega_known=known)
  self.current_state=dict(timestamp_us=int(round(timestamp*1e6)),body_pose=body.tolist(),rear_pose=[*(body[:2]+r),float(body[2])],rear_velocity_body=vr.tolist(),rear_acceleration_body=ar.tolist(),yaw_rate=omega,yaw_acceleration=alpha,yaw_rate_known=known,yaw_acceleration_known=alpha_known,angular_terms_known=known and alpha_known,tire_angle_surrogate=float(np.arctan(3.089*omega/max(abs(float(vr[0])),.5))))
 def route_score(self,raw,props,pose):
  score=np.asarray(raw).copy();ssd=float(score[np.isfinite(score)].std())
  if ssd>1e-6:score=score+ROUTE_ALIGN*ssd*self._route_align_bonus(props,pose)
  return score
 def _replan(self,data,timestamp,pose):
  start=time.perf_counter();v=self.hero_actor.get_velocity();a=self.hero_actor.get_acceleration();cmd=self._command(pose);es={'velocity':[v.x,-v.y],'heading':float(pose[2]),'acceleration':[a.x,-a.y],'command':cmd};img=self._images_tensor(data);ego=model_config.build_ego_status(es,cmd,target_pt=self._route_tp(pose)).unsqueeze(0).unsqueeze(0).to(self.dev)
  with torch.no_grad():out=self.model({'image':img,'ego_status':ego})
  props=out['proposals'][0].float().cpu().numpy();raw=out['pdm_score'][0].float().cpu().numpy();assert props.shape==(64,8,3)and np.isfinite(props).all()and np.isfinite(raw).all();before=self.route_score(raw,props,pose)
  parity=None
  if self.replans==0:
   repeated,_=self.pool_scorer.score(props);parity=float(abs(repeated-raw).max());assert parity<1e-5,parity;assert int(self.route_score(repeated,props,pose).argmax())==int(before.argmax())
  send_packet(self.worker.stdin,dict(variant=self.variant,candidates_body=props,state=self.current_state,rear_offset=self.geometry['rear_offset_math'],route_scores=before,previous_native=self.previous_native,previous_timestamp_us=self.previous_timestamp_us))
  r=read_packet(self.worker.stdout)
  if r is None or not r.get('ok'):raise RuntimeError('pinned execution worker: '+str(r))
  expanded=r['candidates_body'];old_shift=0.;rescore_seconds=0.;logits={}
  if len(expanded)>64:
   t=time.perf_counter();scores,logits=self.pool_scorer.score(expanded);torch.cuda.synchronize();rescore_seconds=time.perf_counter()-t;old_shift=float(abs(scores[:64]-raw).max())
  else:scores=raw
  route_scores=self.route_score(scores,expanded,pose);final=route_scores.copy()
  if self.previous_native is not None:final+=np.log(.01+.99*r['EC'])
  ch=int(final.argmax())
  if self.previous_native is None:assert ch==int(before.argmax()) and len(expanded)==64
  traj=expanded[ch];R=rot(pose[2]);plan=np.zeros((9,3));plan[0]=pose;plan[1:,:2]=pose[:2]+traj[:,:2]@R.T;plan[1:,2]=pose[2]+traj[:,2];self.plan=plan;self.plan_t0=float(timestamp)
  previous=self.previous_native;previous_ts=self.previous_timestamp_us;self.previous_native=r['native'][ch].copy();self.previous_timestamp_us=self.current_state['timestamp_us'];torch.cuda.synchronize();elapsed=time.perf_counter()-start
  rec=dict(replan=self.replans,variant=self.variant,timestamp_us=self.previous_timestamp_us,previous_timestamp_us=previous_ts,choice=ch,original_native_choice=int(before.argmax()),parent=r['parent'],trigger=r['trigger'],variants=r['variants'],original_count=64,pool_count=len(expanded),EC_selected=None if previous is None else float(r['EC'][ch]),original_EC_positive=None if previous is None else int((r['EC'][:64]>0).sum()),offset=r['offset'],offset_rounding_error_seconds=r['offset_rounding_error_seconds'],old_score_max_shift=old_shift,first_same_pool_score_maxerr=parity,body_rear_roundtrip_maxerr_m=r['body_rear_roundtrip_maxerr_m'],execution_seconds=r['seconds'],whole_pool_rescore_seconds=rescore_seconds,replan_seconds=elapsed,current_state=self.current_state,P0=False,ec_lambda=1.,future_observations_used=False)
  np.savez_compressed(self.out/f'transfer_{self.replans:05d}.npz',candidates_body=expanded,candidates_rear=r['candidates_rear'],native=r['native'],EC=r['EC'],EC_available=previous is not None,raw_original_scores=raw,raw_expanded_scores=scores,route_scores=route_scores,final_scores=final,previous_native=np.zeros((0,11))if previous is None else previous,choice=ch,ego_status=ego.detach().cpu().numpy(),current_scene_tokens=self.pool_scorer.context.detach().cpu().numpy(),current_ego_token=self.pool_scorer.ego.detach().cpu().numpy(),**{'logit_'+k:x for k,x in logits.items()})
  with open(self.out/'transfer_plans.jsonl','a')as f:f.write(json.dumps(rec)+'\n')
  if self.replans==0:np.savez_compressed(self.out/'first_sensor_input.npz',**{k:data[k][1]for k in B2D_ORDER})
  self.replans+=1;self.last_plan_seconds=elapsed;self.last_transfer=rec
 def run_step(self,data,timestamp):
  self.observe(float(timestamp));control=super().run_step(data,timestamp);error=None
  if self.previous_native is not None:
   dt=(self.current_state['timestamp_us']-self.previous_timestamp_us)/1e6;pred=np.array([np.interp(dt,np.arange(41)*.1,self.previous_native[:,k])for k in [0,1]]);error=float(np.linalg.norm(pred-np.array(self.current_state['rear_pose'][:2])))
  self.transfer_trace.write(json.dumps(dict(tick=self.tick,state=self.current_state,replans=self.replans,native_tracking_error_m=error,control=dict(steer=control.steer,throttle=control.throttle,brake=control.brake),creep_left=self.creep_left))+'\n');return control
 def destroy(self):
  if hasattr(self,'worker'):
   try:send_packet(self.worker.stdin,dict(close=True));self.worker.wait(timeout=10)
   except Exception:self.worker.terminate();self.worker.wait(timeout=10)
   self.worker_log.close()
  if hasattr(self,'pool_scorer'):self.pool_scorer.close()
  if hasattr(self,'transfer_trace'):self.transfer_trace.close()
  super().destroy()
