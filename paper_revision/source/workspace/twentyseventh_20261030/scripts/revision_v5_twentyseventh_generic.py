"""Three current-PDM-only proposals per fixed generator, no prior-plan geometry.

The existing local perturb_nominal implementation supplies smooth time warp,
lateral onset and curvature offset. Its kinematic flags are diagnostic ONLY:
we use the same dedup + executed EC admission as H3, not an extra generic-only
kinematic filter. Thus this control does not exclude generic candidates more
strictly than H3. All scales and their order are fixed before new labels.
"""
import numpy as np
from coordiworld.candidates import CandidateConfig
from coordiworld.data.pdm_proposer import perturb_nominal
from coordiworld.data.openscene import ego_pose,transform_pose
from nuplan.common.actor_state.state_representation import TimePoint

GENERATORS={
 'Gspeed':dict(speed_scales=[.7,.9,1.075],lateral_offsets_m=[],curvature_offsets_per_m=[]),
 'Gmixed':dict(speed_scales=[.9],lateral_offsets_m=[.75],curvature_offsets_per_m=[-.003]),
}

def proposals(pipe,history):
 pipe.proposer(history)
 trajectory=pipe.proposer.trajectory;times=np.arange(81)*.1
 start=trajectory.start_time.time_us
 states=trajectory.get_state_at_times([TimePoint(start+int(round(t*1e6)))for t in times])
 origin=ego_pose(history[-1]);nominal=np.asarray([transform_pose(np.array([s.rear_axle.x,s.rear_axle.y,s.rear_axle.heading]),origin)for s in states])
 speed=float(np.linalg.norm(history[-1]['ego_dynamic_state'][:2]));out={}
 for name,config in GENERATORS.items():
  cc=CandidateConfig(**config);ca,valid,manifest=perturb_nominal(nominal,times,speed,cc)
  assert ca.shape==(4,8,3)and np.isfinite(ca).all()
  out[name]=[(ca[i].copy(),manifest['variants'][i])for i in range(1,4)]
 return out,dict(source='current-only 8s PDM nominal; no earlier issued-plan geometry',current_timestamp_us=int(history[-1]['timestamp']),nominal_start_us=int(start),current_speed=speed,nominal=nominal)

def duplicate(ca,pool):
 return any(np.linalg.norm(ca[:,:2]-old[:,:2],axis=1).max()<1e-3 and abs(np.arctan2(np.sin(ca[:,2]-old[:,2]),np.cos(ca[:,2]-old[:,2]))).max()<1e-4 for old in pool)
