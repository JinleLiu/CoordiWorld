"""CARLA bridge: execute PDM reference from actual state before M12 perturbation.

The unmodified PDM generator starts on its projected path. Passing that pose
jump to M12's finite-difference gate is invalid for off-centre CARLA starts.
The same pinned controller/actual state is used for Full and BaseEC. Twelve
variants, admissibility thresholds, world and six heads remain unchanged.
"""
import hashlib,json
import numpy as np
from coordiworld.data.openscene import ego_pose
from coordiworld.data.pdm_proposer import perturb_nominal
from coordiworld.revision_v5.compiled_solve_controller import trajectory_array
from coordiworld.revision_v5.compiled_solve_kernel import CompiledPDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from navsim.common.dataclasses import Scene
from navsim.planning.scenario_builder.navsim_scenario_utils import ego_status_to_ego_state
class ControlledNominal:
 def __init__(self,original,capture):
  self.original=original;self.capture=capture;self.planner=original.planner;self.config=original.config;self.sampling=TrajectorySampling(num_poses=80,interval_length=.1);self.sim=CompiledPDMSimulator(self.sampling);self.receipt=None
 def __call__(self,h):
  _,oldvalid,oldmeta=self.original(h)
  initial=ego_status_to_ego_state(Scene._build_ego_status(h[-1]),get_pacifica_parameters(),TimePoint(int(h[-1]['timestamp'])));raw=trajectory_array(self.capture.trajectory,self.sampling,initial.time_point);ex=self.sim.simulate_proposals(raw[None],initial)[0];origin=np.asarray(ego_pose(h[-1]));c,s=np.cos(origin[2]),np.sin(origin[2]);local=ex[:,:3].copy();local[:,:2]=(local[:,:2]-origin[:2])@np.array([[c,-s],[s,c]]);local[:,2]=np.arctan2(np.sin(local[:,2]-origin[2]),np.cos(local[:,2]-origin[2]));np.testing.assert_allclose(local[0],0,atol=1e-7,rtol=0)
  ca,v,meta=perturb_nominal(local,np.arange(81)*.1,float(np.linalg.norm(h[-1]['ego_dynamic_state'][:2])),self.config)
  self.raw=raw;self.executed=ex;self.receipt=dict(version='R18_controlled_nominal_v02',initial_reference_position_displacement_m=float(np.linalg.norm(raw[0,:2]-origin[:2])),initial_reference_heading_displacement_rad=float(np.arctan2(np.sin(raw[0,2]-origin[2]),np.cos(raw[0,2]-origin[2]))),old_valid=int(oldvalid.sum()),new_valid=int(v.sum()),old_fallback=oldmeta['fallback'],label_access=False,raw_reference_sha=hashlib.sha256(raw.tobytes()).hexdigest(),nominal_execution_sha=hashlib.sha256(ex.tobytes()).hexdigest())
  meta['bridge_adapter']=self.receipt;meta.pop('sha256',None);meta['sha256']=hashlib.sha256(ca.tobytes()+v.tobytes()+json.dumps(meta,sort_keys=True).encode()).hexdigest();return ca,v,meta
def install(pipe):
 capture=pipe.proposer;capture.proposer=ControlledNominal(capture.proposer,capture);return pipe
