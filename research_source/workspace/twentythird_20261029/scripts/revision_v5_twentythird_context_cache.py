"""Experimental same-current-input memoization, never cross-frame reuse.
PDM proposer8s and progress-reference5s share proposal4s settings; prefix
equivalence must be checked before promoting this adapter to scored runs.
"""
from revision_v5_twentythird_common import *
import pickle,copy,numpy as np
def key(history):return hashlib.sha256(pickle.dumps(history,protocol=4)).hexdigest()
class CapturedProposer:
 def __init__(self,proposer):
  self.proposer=proposer;self.current_key=None;self.output=None;self.trajectory=None;self.center=None;original=proposer.planner.compute_planner_trajectory
  def capture(inp):
   trajectory=original(inp);self.trajectory=trajectory;self.center=proposer.planner._centerline;return trajectory
  proposer.planner.compute_planner_trajectory=capture
 def __call__(self,history):
  kk=key(history)
  if kk!=self.current_key:
   self.current_key=None;self.output=self.proposer(history);self.current_key=kk
  return copy.deepcopy(self.output)
class ReusedProgress:
 def __init__(self,original,proposer):self.original=original;self.proposer=proposer
 def extract(self,history,assets):
  if self.proposer.current_key!=key(history):return self.original.extract(history,assets)
  from navsim.common.dataclasses import Scene,SensorConfig
  from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
  from navsim.evaluate.pdm_score import get_trajectory_as_array
  from shapely import points
  from coordiworld.data.openscene import ego_pose
  scene=Scene.from_scene_dict_list(history,Path('.'),4,0,SensorConfig.build_no_sensors());sc=NavSimScenario(scene,str(ROOT/'data/maps'),'nuplan-maps-v1.0');planned=get_trajectory_as_array(self.proposer.trajectory,self.original.sampling,sc.initial_ego_state.time_point);executed=self.original.sim.simulate_proposals(planned[None],sc.initial_ego_state)[0];center=self.proposer.center;origin=np.asarray(ego_pose(history[-1]));c,s=np.cos(origin[2]),np.sin(origin[2]);ex=assets['execution_full_local'].copy();ex[:,:,:2]=ex[:,:,:2]@np.array([[c,s],[-s,c]])+origin[:2];ex[:,:,2]+=origin[2];offset=sc.initial_ego_state.car_footprint.vehicle_parameters.rear_axle_to_center
  def progress(states):
   xy=states[...,:2]+offset*np.stack([np.cos(states[...,2]),np.sin(states[...,2])],-1);arc=np.asarray(center.project(points(xy[...,0],xy[...,1])));return np.maximum(arc[...,-1]-arc[...,0],0),center.length-arc[...,-1],arc
  d,rem,arc=progress(ex);db,rb,barc=progress(executed)
  return dict(distance=d,pdm_distance=float(db),remaining=rem,pdm_planned=planned,pdm_executed=executed,candidate_executed=ex,centerline=np.array([[x.x,x.y,x.heading]for x in center.discrete_path]),candidate_arc=arc,pdm_arc=barc),dict(current_only=True,current_timestamp_us=int(history[-1]['timestamp']),history_frames=4,future_frames=0,reused_current_input_key=self.proposer.current_key,PDM_initialization='samecurrent proposer trajectory first4s; experimental prefixequivalence adapter',rear_axle_to_center=offset,centerline_length_m=float(center.length))
