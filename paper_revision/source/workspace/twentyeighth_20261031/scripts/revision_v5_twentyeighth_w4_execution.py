"""Isolated pinned NAVSIM execution/EC for W4S; never imports its NAVSIM fork."""
import sys,struct,pickle,traceback,json,hashlib
from pathlib import Path
import numpy as np
def wrap(x):return np.arctan2(np.sin(x),np.cos(x))
def rotate_offset(yaw,offset):
 return np.stack([np.cos(yaw)*offset[0]-np.sin(yaw)*offset[1],np.sin(yaw)*offset[0]+np.cos(yaw)*offset[1]],-1)
def body_to_rear(c,off):
 z=np.asarray(c,np.float64).copy();z[...,:2]+=rotate_offset(z[...,2],off)-off;return z.astype(np.float32)
def rear_to_body(c,off):
 z=np.asarray(c,np.float64).copy();z[...,:2]-=rotate_offset(z[...,2],off)-off;return z.astype(np.float32)
class Executor:
 def __init__(self):
  from coordiworld.revision_v5.compiled_solve_kernel import CompiledPDMSimulator
  from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
  self.sampling=TrajectorySampling(num_poses=40,interval_length=.1);self.sim=CompiledPDMSimulator(self.sampling)
 def initial(self,state):
  from nuplan.common.actor_state.ego_state import EgoState
  from nuplan.common.actor_state.state_representation import StateSE2,StateVector2D,TimePoint
  from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
  return EgoState.build_from_rear_axle(StateSE2(*state['rear_pose']),StateVector2D(*state['rear_velocity_body']),StateVector2D(*state['rear_acceleration_body']),state['tire_angle_surrogate'],TimePoint(state['timestamp_us']),get_pacifica_parameters(),angular_vel=state['yaw_rate'],angular_accel=state['yaw_acceleration'])
 def execute(self,candidates,initial):
  from coordiworld.revision_v5.compiled_solve_controller import numeric_trajectory,trajectory_array
  raw=np.stack([trajectory_array(numeric_trajectory(c,initial),self.sampling,initial.time_point)for c in candidates]);return self.sim.simulate_proposals(raw,initial)
 def process(self,r):
  import time
  from revision_v5_twentythird_factory import continuation,ec_matrix,extend
  from revision_v5_twentythird_spatial_probe import spatial_dynamic
  from revision_v5_twentythird_null_probe import current_profile_candidate
  from revision_v5_twentyseventh_generic import duplicate
  from coordiworld.candidates import CandidateConfig
  from coordiworld.data.pdm_proposer import perturb_nominal
  t=time.perf_counter();body=np.array(r['candidates_body'],np.float32,copy=True);assert body.shape==(64,8,3)and np.isfinite(body).all();offset=np.array(r['rear_offset']);rear=body_to_rear(body,offset);roundtrip=rear_to_body(rear,offset);err=float(abs(body[...,:2]-roundtrip[...,:2]).max());assert err<2e-5,err
  initial=self.initial(r['state']);native=self.execute(rear,initial);ts=r['state']['timestamp_us'];prior=r.get('previous_native');pts=r.get('previous_timestamp_us');off=None;ec=np.ones(64);variants=[];trigger=False;original_native=native.copy()
  if prior is not None:
   off=round((ts-pts)/1e5);assert 0<off<len(prior)-3;ec=ec_matrix(native,prior,off)
  anchor_scores=np.asarray(r['route_scores'])+(0 if off is None else np.log(.01+.99*ec));parent=int(anchor_scores.argmax())
  if off is not None and r['variant']in ['EC_H3','EC_Gspeed']:
   trigger=not bool((ec>0).any())
   if trigger:
    generated=[]
    if r['variant']=='EC_H3':
     ca,meta=continuation(native[0,0],prior,ts,pts,2.,'dynamic');generated.append(('history',ca,meta))
     ca,meta=current_profile_candidate(native,parent,ts,pts);generated.append(('current',ca,meta))
     ca,meta=spatial_dynamic(native[0,0],prior,off);generated.append(('spatial',ca,meta))
    else:
     # Only the currently predicted parent execution supplies this nominal.
     # Its missing4s tail is explicit terminal-curvature/decaying-a extrapolation.
     tail,extension=extend(native[parent],40);nom=np.concatenate([native[parent],tail[1:]])[:,:3];origin=native[parent,0,:3];c,s=np.cos(origin[2]),np.sin(origin[2]);nom[:,:2]=(nom[:,:2]-origin[:2])@np.array([[c,-s],[s,c]]);nom[:,2]=wrap(nom[:,2]-origin[2]);cc=CandidateConfig(speed_scales=[.7,.9,1.075],lateral_offsets_m=[],curvature_offsets_per_m=[]);ca,v,meta=perturb_nominal(nom,np.arange(81)*.1,max(0,float(native[parent,0,3])),cc)
     for j,scale in enumerate([.7,.9,1.075],1):generated.append(('speed_'+str(scale),ca[j],dict(scale=scale,nominal_source='current W4S+EC parent predicted execution, extended4s; no historical plan geometry',extension=extension,kinematic_flag_diagnostic_only=bool(v[j]),perturbation=meta['variants'][j])))
    for name,ca,meta in generated:
     vr=dict(name=name,admitted=False,detail=meta,duplicate=duplicate(ca,rear))
     if not vr['duplicate']:
      ex=self.execute(ca[None],initial);ce=float(ec_matrix(ex,prior,off)[0]);vr['EC']=ce
      if ce>0:
       vr.update(admitted=True,index=len(body));rear=np.concatenate([rear,ca[None]]);body=np.concatenate([body,rear_to_body(ca[None],offset)]);native=np.concatenate([native,ex]);ec=np.r_[ec,ce]
     variants.append(vr)
  assert np.array_equal(body[:64],r['candidates_body']);return dict(ok=True,candidates_body=body,candidates_rear=rear,native=native,EC=ec,parent=parent,trigger=trigger,variants=variants,offset=off,offset_rounding_error_seconds=None if off is None else off*.1-(ts-pts)/1e6,body_rear_roundtrip_maxerr_m=err,original_native_sha=hashlib.sha256(original_native.tobytes()).hexdigest(),seconds=time.perf_counter()-t,execution_model='pinned NAVSIM compiled Pacifica LQR/bicycle; not CARLA actual future',prior_source='own previously selected predicted execution',future_observations_used=False)
def read_packet(stream):
 h=stream.read(8)
 if not h:return None
 n=struct.unpack('!Q',h)[0];data=bytearray()
 while len(data)<n:
  part=stream.read(n-len(data))
  if not part:raise EOFError('truncated internal packet')
  data.extend(part)
 return pickle.loads(data)
def send_packet(stream,r):
 b=pickle.dumps(r,protocol=5);stream.write(struct.pack('!Q',len(b)));stream.write(b);stream.flush()
if __name__=='__main__':
 wire=sys.stdout.buffer;sys.stdout=sys.stderr;engine=Executor()
 while True:
  q=read_packet(sys.stdin.buffer)
  if q is None or q.get('close'):break
  try:out=engine.process(q)
  except Exception:out=dict(ok=False,error=traceback.format_exc())
  send_packet(wire,out)
