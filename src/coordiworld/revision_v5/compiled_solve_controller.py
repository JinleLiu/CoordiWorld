"""Pinned-controller acceleration: avoid rebuilding 41 EgoState objects/action.

Retains native pose conversion, time arithmetic, interpolation, and simulator.
The private interpolation attributes are version-pinned and tested end to end.
"""
import numpy as np
from .native_adapter import global_execution_to_local


def trajectory_array(trajectory,sampling,start_time):
    times=np.arange(0.,sampling.time_horizon+sampling.interval_length,sampling.interval_length)
    times+=start_time.time_s
    micros=np.clip(np.asarray([int(t*1e6) for t in times]),trajectory.start_time.time_us,trajectory.end_time.time_us)
    linear=trajectory._function_interp_linear(micros);angular=trajectory._angular_interpolator.interpolate(micros)
    result=np.zeros((len(times),11),np.float64)
    result[:,:2]=linear[:,1:3];result[:,2]=angular[:,0];result[:,3:7]=linear[:,3:7];result[:,7]=linear[:,7]
    return result



def numeric_trajectory(candidates,initial):
    """Same pinned split-state interpolation without future EgoState allocation."""
    from types import SimpleNamespace
    from scipy.interpolate import interp1d
    from nuplan.common.actor_state.state_representation import StateSE2,TimePoint
    from nuplan.common.geometry.convert import relative_to_absolute_poses
    from nuplan.common.geometry.compute import AngularInterpolator
    from nuplan.planning.simulation.planner.ml_planner.transform_utils import _get_fixed_timesteps
    times=[initial.time_us]+[int(t*1e6) for t in _get_fixed_timesteps(initial,4.,.5)]
    absolute=relative_to_absolute_poses(initial.rear_axle,[StateSE2.deserialize(p) for p in candidates.astype(np.float64)])
    split=initial.to_split_state();linear=np.zeros((9,8),np.float64);angular=np.zeros((9,1),np.float64)
    linear[0]=split.linear_states;angular[0]=split.angular_states
    for k,pose in enumerate(absolute,1):
        linear[k,:3]=[times[k],pose.x,pose.y];angular[k,0]=pose.heading
    return SimpleNamespace(start_time=TimePoint(times[0]),end_time=TimePoint(times[-1]),_function_interp_linear=interp1d(times,linear,axis=0),_angular_interpolator=AngularInterpolator(times,angular))

def predict_vector_controller_execution(history,candidates,candidate_valid):
    candidates=np.asarray(candidates);cv=np.asarray(candidate_valid)
    if len(history)!=4 or candidates.shape[1:]!=(8,3) or cv.dtype!=np.bool_ or not cv.any():raise ValueError('Invalid current controller inputs')
    times=np.array([f['timestamp'] for f in history],dtype=np.float64)
    if not np.allclose(np.diff(times)/1e6,.5,atol=.08):raise ValueError('Noncontiguous history')
    from navsim.common.dataclasses import Scene,Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory
    from navsim.planning.scenario_builder.navsim_scenario_utils import ego_status_to_ego_state
    from .compiled_solve_kernel import CompiledPDMSimulator as PDMSimulator
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    initial=ego_status_to_ego_state(Scene._build_ego_status(history[-1]),get_pacifica_parameters(),TimePoint(int(times[-1])))
    sampling=TrajectorySampling(num_poses=40,interval_length=.1);ps=TrajectorySampling(num_poses=8,interval_length=.5)
    ids=np.flatnonzero(cv);proposals=[]
    for j in ids:
        trajectory=numeric_trajectory(candidates[j],initial)
        proposals.append(trajectory_array(trajectory,sampling,initial.time_point))
    executed=PDMSimulator(sampling).simulate_proposals(np.stack(proposals),initial)
    origin=np.array([initial.rear_axle.x,initial.rear_axle.y,initial.rear_axle.heading]);local=np.zeros_like(candidates,dtype=np.float32);full={}
    for j,value in zip(ids,executed):local[j]=global_execution_to_local(value,origin);full[int(j)]=value
    return local,full,{'input_latest_timestamp':int(times[-1]),'num_future_frames':0,'uses_metric_cache':False,'uses_traffic_teacher':False,'native_proposals_batched':True,'vector_interpolation':True,'numeric_future_state_construction':True,'triangular_LQR_prefix_sums':True,'compiled_float64_control_loop':True,'profile_full_rank_direct_solve':True,'coordinate_origin':origin.tolist()}
