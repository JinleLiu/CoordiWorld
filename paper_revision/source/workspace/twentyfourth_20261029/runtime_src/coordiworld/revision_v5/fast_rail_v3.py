"""Current-only lane rails and parallel follower dynamics with projected follower occupancy and curved ego-path eligibility; v3 surrogate.

Uses static current-segment paths, matching the frozen v2 teacher domain. The
parallel rectangular leader approximation differs from native sequential exact
polygon occupancy. Neural distillation must beat its own untrained controls.
"""
import numpy as np
import torch
from torch import nn


def extract_rails(context):
    from copy import deepcopy
    from .current_rail_response import make_policy
    from navsim.traffic_agents_policies.abstract_traffic_agents_policy import filter_tracked_objects_by_type
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    p=context.map_parameters;mp=get_maps_api(p.map_root,p.map_version,p.map_name)
    vehicle=filter_tracked_objects_by_type(context.current_tracked_objects,TrackedObjectType.VEHICLE)[0]
    obs=make_policy(p.map_root)._idm_agents_observation
    manager=obs._get_idm_agent_manager(context.ego_state,vehicle,mp)
    origin=context.ego_state.rear_axle;c,s=np.cos(origin.heading),np.sin(origin.heading);rot=np.array([[c,s],[-s,c]])
    tokens=[];paths=[];info=[]
    for token,a in manager.agents.items():
        # Preserve every active current actor, including relevant outside-ROI leaders.
        q=a._path;start,end=q.get_start_progress(),q.get_end_progress()
        if not end>start:raise ValueError('Degenerate rail')
        progress=np.linspace(start,end,256);poses=q.get_state_at_progresses(progress.tolist())
        xy=np.array([[v.x,v.y] for v in poses]);xy=(xy-np.array([origin.x,origin.y]))@rot.T
        heading=np.unwrap(np.array([v.heading-origin.heading for v in poses]))
        paths.append(np.c_[xy,heading]);tokens.append(token)
        info.append([a.progress-start,end-start,a.velocity,a.length,a.width,a._policy.target_velocity,float(a.velocity<=.5),len(a.end_segment.outgoing_edges)])
    return np.asarray(paths,np.float32).reshape(-1,256,3),np.asarray(info,np.float32).reshape(-1,8),tokens


def interpolate(paths,progress,length):
    # Uniform samples include both exact native path endpoints; no future map.
    u=(progress/length.clamp_min(1e-4)[None]*255).clamp(0,255);lo=u.long().clamp_max(254);f=u-lo
    ids=torch.arange(paths.shape[0],device=paths.device)[None].expand(progress.shape)
    a=paths[ids,lo];b=paths[ids,lo+1]
    return a+(b-a)*f[...,None]


def ego_path_contact(paths, info, ego, ego_length=5.176, ego_width=2.297):
    """Current static path segments against each candidate ego footprint.

    Segment/expanded-box slab intersection is a conservative approximation to
    the native buffered curved polyline. It replaces the infinite tangent test;
    this is NOT exact native polygon intersection or a learned interaction.
    Returns last intersecting progress, or -infinity when no contact exists.
    """
    # M x T x N x path samples. No teacher and no per-scene prediction cache.
    relative = paths[None,None,:,:,:2] - ego[:,:,None,None,:2]
    c = ego[:,:,2].cos()[:,:,None,None]
    s = ego[:,:,2].sin()[:,:,None,None]
    local_x = relative[...,0]*c + relative[...,1]*s
    local_y = -relative[...,0]*s + relative[...,1]*c
    radius = info[:,4][None,None,:,None]/2
    half_x = ego_length/2 + radius
    half_y = ego_width/2 + radius
    lower = torch.zeros_like(local_x[...,:-1])
    upper = torch.ones_like(lower)
    valid = torch.ones_like(lower, dtype=torch.bool)
    for coordinate, half in [(local_x,half_x),(local_y,half_y)]:
        p = coordinate[...,:-1]; d = coordinate[...,1:]-p
        parallel = d.abs()<1e-8
        divisor = torch.where(parallel,torch.ones_like(d),d)
        a=(-half-p)/divisor; b=(half-p)/divisor
        lo=torch.minimum(a,b); hi=torch.maximum(a,b)
        lower=torch.maximum(lower,torch.where(parallel,torch.full_like(lo,-float('inf')),lo))
        upper=torch.minimum(upper,torch.where(parallel,torch.full_like(hi,float('inf')),hi))
        valid=valid&(~parallel|(p.abs()<=half))
    valid=valid&(lower<=upper)
    idx=torch.arange(255,device=paths.device,dtype=paths.dtype)
    progress=(idx+upper.clamp(0,1))*info[:,1][None,None,:,None]/255
    return progress.masked_fill(~valid,-float('inf')).amax(-1)


def fast_rollout(paths,info,ego,ego_length=5.176,ego_width=2.297):
    """Batched M actions, 40x0.1s; outputs half-second states for active rails.

Ego arrays are corrected body-center poses and actual controller speed; actor
progress/speed are integrated together including stopped vehicles. No teacher.
"""
    m=ego.shape[0];n=info.shape[0]
    if n==0:return paths.new_zeros(m,8,0,7)
    progress=info[:,0][None].expand(m,-1).clone();speed=info[:,2][None].expand(m,-1).clone()
    end=info[:,1];target=info[:,5].clamp_min(.1);length=info[:,3];width=info[:,4]
    eye=torch.eye(n,dtype=torch.bool,device=paths.device)[None];outputs=[]
    ego_contact=ego_path_contact(paths,info,ego,ego_length,ego_width)
    alive=torch.ones(m,n,dtype=torch.bool,device=paths.device)
    for k in range(40):
        pose=interpolate(paths,progress,end);xy=pose[...,:2];yaw=pose[...,2];c,s=yaw.cos(),yaw.sin()
        # Projected leader boxes include forward speed*1.5 headway, as in native.
        extension=torch.minimum(speed*1.5,(end[None]-progress-length[None]/2).clamp_min(0))
        own_extension=extension if k>0 else torch.zeros_like(extension)
        jxy=xy+.5*extension[...,None]*torch.stack((c,s),-1)
        rel=jxy[:,None]-xy[:,:,None]
        lon=rel[...,0]*c[:,:,None]+rel[...,1]*s[:,:,None]
        lat=-rel[...,0]*s[:,:,None]+rel[...,1]*c[:,:,None]
        angle=yaw[:,None]-yaw[:,:,None];ca,sa=angle.cos(),angle.sin()
        pair_extension=extension[:,None].expand(-1,n,-1)
        if k==0:
            earlier=torch.arange(n,device=paths.device)[None,:]<torch.arange(n,device=paths.device)[:,None]
            pair_extension=pair_extension*earlier[None]
            offset=(pair_extension-extension[:,None])*.5
            lon=lon+offset*ca;lat=lat+offset*sa
        half_len=(length[None,None]+pair_extension)/2
        ex=half_len*ca.abs()+width[None,None]/2*sa.abs();ey=half_len*sa.abs()+width[None,None]/2*ca.abs()
        remaining=end[None]-progress
        eligible=(lon+ex>=0)&(lon-ex<=remaining[:,:,None])&(lat.abs()<=width[None,:,None]/2+ey)&~eye&alive[:,None]
        distance_lon=lon-.5*own_extension[:,:,None]
        dx=(distance_lon.abs()-(length[None,:,None]+own_extension[:,:,None])/2-ex).clamp_min(0);dy=(lat.abs()-width[None,:,None]/2-ey).clamp_min(0)
        gap=torch.sqrt(dx.square()+dy.square())
        gap=gap.masked_fill(~eligible,float('inf'));nearest,index=gap.min(-1)
        lead=torch.gather(speed,1,index)*torch.gather(ca,2,index[...,None]).squeeze(-1)
        # Ego is a normal box, without an artificial projected headway extension.
        e=ego[:,k];er=e[:,None,:2]-xy;el=er[...,0]*c+er[...,1]*s;et=-er[...,0]*s+er[...,1]*c
        ec=(e[:,None,2]-yaw).cos();es=(e[:,None,2]-yaw).sin()
        ex0=ego_length/2*ec.abs()+ego_width/2*es.abs();ey0=ego_length/2*es.abs()+ego_width/2*ec.abs()
        egap=torch.sqrt(((el-.5*own_extension).abs()-(length[None]+own_extension)/2-ex0).clamp_min(0).square()+(et.abs()-width[None]/2-ey0).clamp_min(0).square())
        possible=ego_contact[:,k]>=progress
        egap=egap.masked_fill(~possible,float('inf'));useego=egap<=nearest
        nearest=torch.minimum(egap,nearest);lead=torch.where(useego,e[:,None,3]*ec,lead)
        free=~torch.isfinite(nearest);distance=torch.where(free,remaining-length[None]/2,nearest).clamp_min(1.)
        lead=torch.where(free,0.,lead)
        desired=1+speed*1.5+speed*(speed-lead)/(2*np.sqrt(2.))
        accel=(1-(speed/target[None]).pow(4)-(desired/distance).square()).clamp(-2,1)
        alive=alive&(torch.linalg.vector_norm(xy-e[:,None,:2],dim=-1)<=100)
        progress=progress+speed*.1*alive;speed=torch.where(alive,(speed+accel*.1).clamp_min(0),speed)
        if (k+1)%5==0:
            p=interpolate(paths,progress,end);velocity=speed[...,None]*torch.stack((p[...,2].cos(),p[...,2].sin()),-1)
            outputs.append(torch.cat((p,velocity,length[None,:,None].expand(m,-1,-1),width[None,:,None].expand(m,-1,-1)),-1))
    return torch.stack(outputs,1)


def map_to_slots(states,slot_indices,nslots):
    out=states.new_zeros(states.shape[0],8,nslots,7);good=slot_indices>=0
    if good.any():out[:,:,slot_indices[good]]=states[:,:,good]
    return out


def distilled_features(states,info,ego,reference):
    """One sample per candidate x actor, full horizon plus lane and leader cues."""
    m,h,n,_=states.shape;ref=states[reference:reference+1];d=states[...,:2]-ref[...,:2]
    yaw=states[reference,0,:,2];c,s=yaw.cos()[None,None],yaw.sin()[None,None]
    local=torch.stack((d[...,0]*c+d[...,1]*s,-d[...,0]*s+d[...,1]*c),-1)
    er=ego[:,4::5,None,:2]-ref[...,:2]
    relative=torch.stack((er[...,0]*c+er[...,1]*s,-er[...,0]*s+er[...,1]*c),-1)/20
    ed=relative-relative[reference:reference+1]
    temporal=torch.cat((local,relative,ed),-1).permute(0,2,1,3).flatten(2)
    static=info.clone();static[:,0]/=50;static[:,1]/=100;static[:,2]/=20;static[:,3:5]/=10;static[:,5]/=20;static[:,7]/=4
    maxdelta=torch.linalg.vector_norm(d,dim=-1).amax(1)[...,None]
    features=torch.cat((temporal,static[None].expand(m,-1,-1),maxdelta),-1).clamp(-20,20)
    return features,local,yaw


class RailDistiller(nn.Module):
    def __init__(self,input_dim,mode):
        super().__init__();self.mode=mode
        self.net=nn.Sequential(nn.Linear(input_dim,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1 if mode=='gate' else 16))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,features,prior_local,yaw,reference):
        z=self.net(features)
        if self.mode=='gate':local=prior_local*(2*torch.sigmoid(z))[:,None]
        else:
            # Direct learned displacement, bounded by the current physical scale.
            scale=torch.linalg.vector_norm(prior_local,dim=-1).amax(1).clamp_min(.02).clamp_max(5.)
            local=2*torch.tanh(z.reshape(z.shape[0],z.shape[1],8,2).permute(0,2,1,3))*scale[:,None,:,None]
        local=local-local[reference:reference+1]
        c,s=yaw.cos()[None,None],yaw.sin()[None,None]
        return torch.stack((local[...,0]*c-local[...,1]*s,local[...,0]*s+local[...,1]*c),-1)
