"""Route-aware frozen proposer using current observations only.

This is a new, named replacement for the paper's unspecified nominal proposer.
It never receives the future frames used by the metric-cache/scoring pipeline.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from ..candidates import CandidateConfig
from .openscene import ego_pose,transform_pose


def perturb_nominal(nominal,times,current_speed,config):
    output_times=np.arange(1,9)*.5
    variants=[('nominal',1.,0.,0.)]+[(f'speed_{s}',s,0.,0.) for s in config.speed_scales]+[(f'lateral_{d}',1.,d,0.) for d in config.lateral_offsets_m]+[(f'curvature_{k}',1.,0.,k) for k in config.curvature_offsets_per_m]
    nominal=nominal.copy();nominal[:,2]=np.unwrap(nominal[:,2])
    outputs=[];valid=[];diagnostics=[]
    for name,scale,offset,curvature in variants:
        # Integrate a smooth transition of time scaling; initial velocity is
        # preserved instead of abruptly multiplying all velocities at t=0.
        t=output_times;transition=2.;u=np.minimum(t/transition,1.)
        integral=np.where(t<transition,transition*(u**3-.5*u**4),t-transition/2)
        query=t+(scale-1)*integral
        if query[-1]>times[-1]:raise ValueError('Nominal trajectory horizon too short')
        poses=np.stack([np.interp(query,times,nominal[:,j]) for j in range(3)],-1)
        if offset or curvature:
            distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(nominal[:,:2],axis=0),axis=-1))]
            arc=np.interp(query,times,distance)
            phase=t/t[-1];blend=10*phase**3-15*phase**4+6*phase**5
            displacement=offset*blend+.5*curvature*arc**2
            poses[:,:2]+=displacement[:,None]*np.stack((-np.sin(poses[:,2]),np.cos(poses[:,2])),-1)
            delta=np.diff(np.vstack(([0.,0.],poses[:,:2])),axis=0)
            moving=np.linalg.norm(delta,axis=-1)>1e-5
            poses[:,2]=np.where(moving,np.arctan2(delta[:,1],delta[:,0]),poses[:,2])
        full=np.vstack(([0.,0.,0.],poses))
        speed=np.linalg.norm(np.diff(full[:,:2],axis=0),axis=-1)/.5
        acceleration=np.diff(np.r_[current_speed,speed])/.5
        yaw_rate=np.diff(np.unwrap(full[:,2]))/.5
        admissible=bool(acceleration.max()<=config.max_acceleration+.15 and acceleration.min()>=-config.max_deceleration-.15 and abs(yaw_rate).max()<=config.max_yaw_rate+.05 and (speed*abs(yaw_rate)).max()<=config.max_lateral_acceleration+.2)
        duplicate=any(np.linalg.norm(poses[:,:2]-prior[:,:2],axis=-1).max()<config.duplicate_tolerance_m for prior,ok in zip(outputs,valid) if ok)
        outputs.append(poses);valid.append(admissible and not duplicate)
        diagnostics.append({'name':name,'kinematically_admissible':admissible,'duplicate':duplicate})
    fallback=None
    if not any(valid):
        # Keep the scene in the denominator. An explicitly recorded emergency
        # braking proposal replaces slot zero; do not discard hard scenes.
        v=float(current_speed);distance=0.;poses=[]
        for _ in output_times:
            next_v=max(0.,v-config.max_deceleration*.5)
            distance+=(v+next_v)*.25;poses.append([distance,0.,0.]);v=next_v
        outputs[0]=np.asarray(poses);valid[0]=True
        fallback='constant_heading_emergency_brake_no_admissible_pdm_variants'
        diagnostics[0]={'name':'emergency_brake_fallback','kinematically_admissible':True,'duplicate':False}
    trajectories=np.asarray(outputs,np.float32);valid=np.asarray(valid,bool)
    manifest={'generator':'pdm_closed_current_v1','dt':.5,'steps':8,'variants':diagnostics,'config':vars(config),'nominal_sampling':{'poses':80,'dt':.1},'input_scope':'current ego, observed tracks, current signals, static map and given route; no future frames'}
    manifest['fallback']=fallback
    manifest['sha256']=hashlib.sha256(trajectories.tobytes()+valid.tobytes()+json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    return trajectories,valid,manifest


class PDMNominalProposer:
    def __init__(self,map_root,config):
        os.environ['NUPLAN_MAPS_ROOT']=str(Path(map_root).resolve())
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
        from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
        self.map_root=str(Path(map_root).resolve());self.config=CandidateConfig(**config)
        self.planner=PDMClosedPlanner(TrajectorySampling(num_poses=80,interval_length=.1),TrajectorySampling(num_poses=40,interval_length=.1),BatchIDMPolicy(speed_limit_fraction=[.2,.4,.6,.8,1.],fallback_target_velocity=15.,min_gap_to_lead_agent=1.,headway_time=1.5,accel_max=1.5,decel_max=3.),[-1.,1.],100.)

    def __call__(self,history):
        from nuplan.common.actor_state.state_representation import TimePoint
        from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
        from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization,PlannerInput
        from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
        from navsim.common.dataclasses import Scene,SensorConfig
        from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
        scene=Scene.from_scene_dict_list(history,Path('.'),len(history),0,SensorConfig.build_no_sensors())
        scenario=NavSimScenario(scene,self.map_root,'nuplan-maps-v1.0')
        current=scenario.initial_ego_state
        initialization=PlannerInitialization(scenario.get_route_roadblock_ids(),scenario.get_mission_goal(),scenario.map_api)
        buffer=SimulationHistoryBuffer.initialize_from_list(1,[current],[scenario.initial_tracked_objects])
        planner_input=PlannerInput(SimulationIteration(current.time_point,0),buffer,list(scenario.get_traffic_light_status_at_iteration(0)))
        self.planner.initialize(initialization)
        try:trajectory=self.planner.compute_planner_trajectory(planner_input)
        finally:gc.enable()  # Upstream planner disables cyclic GC during simulation.
        times=np.arange(81)*.1
        states=trajectory.get_state_at_times([TimePoint(current.time_point.time_us+int(round(t*1e6))) for t in times])
        origin=ego_pose(history[-1]);poses=np.array([transform_pose(np.array([s.rear_axle.x,s.rear_axle.y,s.rear_axle.heading]),origin) for s in states])
        speed=float(np.linalg.norm(history[-1]['ego_dynamic_state'][:2]))
        return perturb_nominal(poses,times,speed,self.config)
