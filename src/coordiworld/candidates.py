"""Deterministic current-state candidate proposer."""
from dataclasses import dataclass, field
import hashlib
import json
import numpy as np


@dataclass
class CandidateConfig:
    nominal: str = 'current_state_ctrv_v1'
    speed_scales: list = field(default_factory=lambda:[0.60,0.80,1.15])
    lateral_offsets_m: list = field(default_factory=lambda:[-1.0,-0.5,0.5,1.0])
    curvature_offsets_per_m: list = field(default_factory=lambda:[-0.004,-0.002,0.002,0.004])
    max_acceleration: float = 3.0
    max_deceleration: float = 6.0
    max_lateral_acceleration: float = 4.0
    max_yaw_rate: float = 0.5
    duplicate_tolerance_m: float = 0.001


def generate_candidates(speed, yaw_rate, steps=8, dt=0.5, config=None):
    """Return fixed slots, an admissibility mask and explicit diagnostics.

    The nominal maneuver is CTRV, with no route or future access. Variants share
    the same current pose. Duplicate/infeasible slots remain serialized but are
    excluded for every evaluator. No claim of route feasibility is made.
    """
    config=config or CandidateConfig()
    if config.nominal!='current_state_ctrv_v1':raise ValueError(config.nominal)
    speed=max(float(speed),0.0)
    yaw_rate=float(np.clip(yaw_rate,-config.max_yaw_rate,config.max_yaw_rate))
    base_curvature=np.clip(yaw_rate/max(speed,0.5),-config.max_lateral_acceleration/max(speed**2,1),config.max_lateral_acceleration/max(speed**2,1))
    variants=[('nominal',1.0,0.0,0.0)]
    variants += [(f'speed_{s}',s,0.0,0.0) for s in config.speed_scales]
    variants += [(f'lateral_{d}',1.0,d,0.0) for d in config.lateral_offsets_m]
    variants += [(f'curvature_{k}',1.0,0.0,k) for k in config.curvature_offsets_per_m]
    times=np.arange(1,steps+1)*dt
    outputs=[];valid=[];diagnostics=[]
    for name,scale,offset,curvature_delta in variants:
        v=speed;x=0.0;y=0.0;yaw=0.0;poses=[];velocities=[]
        curvature=base_curvature+curvature_delta
        for _ in times:
            new_v=float(np.clip(speed*scale,v-config.max_deceleration*dt,v+config.max_acceleration*dt))
            travel=(v+new_v)*dt/2
            angle=travel*curvature
            if abs(curvature)>1e-9:
                x+=(np.sin(yaw+angle)-np.sin(yaw))/curvature
                y+=(-np.cos(yaw+angle)+np.cos(yaw))/curvature
            else:x+=travel*np.cos(yaw);y+=travel*np.sin(yaw)
            yaw+=angle;v=new_v;poses.append([x,y,yaw]);velocities.append(v)
        poses=np.asarray(poses)
        if offset:
            # Quintic onset: zero lateral displacement/velocity/acceleration at t=0.
            u=times/times[-1];blend=10*u**3-15*u**4+6*u**5
            tangent=poses[:,2].copy()
            poses[:,:2]+=offset*blend[:,None]*np.stack((-np.sin(tangent),np.cos(tangent)),-1)
            full=np.vstack(([0,0,0],poses))
            delta=np.diff(full[:,:2],axis=0)
            headings=np.arctan2(delta[:,1],delta[:,0])
            poses[:,2]=np.where(np.linalg.norm(delta,axis=-1)>1e-5,headings,0.0)
        full=np.vstack(([0,0,0],poses))
        vxy=np.diff(full[:,:2],axis=0)/dt
        speeds=np.linalg.norm(vxy,axis=-1)
        acceleration=np.diff(np.r_[speed,speeds])/dt
        rates=np.diff(np.unwrap(full[:,2]))/dt
        admissible=bool(np.max(acceleration)<=config.max_acceleration+0.15 and np.min(acceleration)>=-config.max_deceleration-0.15 and np.max(abs(rates))<=config.max_yaw_rate+0.05 and np.max(speeds*abs(rates))<=config.max_lateral_acceleration+0.2)
        duplicate=any(np.max(np.linalg.norm(poses[:,:2]-prior[:,:2],axis=-1))<config.duplicate_tolerance_m for prior,ok in zip(outputs,valid) if ok)
        outputs.append(poses);valid.append(admissible and not duplicate)
        diagnostics.append({'name':name,'kinematically_admissible':admissible,'duplicate':duplicate})
    if not any(valid):raise ValueError('No admissible candidate; scene must be recorded as proposer failure')
    trajectories=np.stack(outputs).astype(np.float32)
    manifest={'generator':config.nominal,'dt':dt,'steps':steps,'variants':diagnostics,'config':vars(config)}
    manifest['sha256']=hashlib.sha256(trajectories.tobytes()+np.asarray(valid,dtype=np.bool_).tobytes()+json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    return trajectories,np.asarray(valid),manifest
