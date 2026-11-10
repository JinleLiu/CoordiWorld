"""Geometry uses explicit oriented boxes; no center-distance collision labels."""
import numpy as np
import torch
from torch import Tensor


def wrap_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x)) if isinstance(x,Tensor) else np.arctan2(np.sin(x),np.cos(x))


def rear_axle_to_box(poses, length_width, offset):
    """Convert (...,3) rear-axle poses into (...,5) x/y/yaw/length/width boxes."""
    center = poses[...,:2] + offset[...,None] * torch.stack((poses[...,2].cos(),poses[...,2].sin()),-1)
    return torch.cat((center,poses[...,2:3],length_width.expand(*poses.shape[:-1],2)),-1)


def sat_clearance(a, b):
    """Signed separating-axis gap of broadcastable (...,5) OBBs.

    Positive = disjoint, <=0 = overlap/touch. This is a SAT margin rather than
    Euclidean distance. Using max of four gaps gives an exact overlap predicate.
    """
    ca,sa,cb,sb = a[...,2].cos(),a[...,2].sin(),b[...,2].cos(),b[...,2].sin()
    ua,va = torch.stack((ca,sa),-1),torch.stack((-sa,ca),-1)
    ub,vb = torch.stack((cb,sb),-1),torch.stack((-sb,cb),-1)
    delta = b[...,:2] - a[...,:2]
    gaps=[]
    for axis in (ua,va,ub,vb):
        ra=(axis*ua).sum(-1).abs()*a[...,3]/2+(axis*va).sum(-1).abs()*a[...,4]/2
        rb=(axis*ub).sum(-1).abs()*b[...,3]/2+(axis*vb).sum(-1).abs()*b[...,4]/2
        gaps.append((delta*axis).sum(-1).abs()-ra-rb)
    return torch.stack(gaps,-1).amax(-1)


def collision_labels(inputs, future_agents, future_valid, candidates=None):
    """Non-reactive logged-box replay, deliberately NOT official NAVSIM NC."""
    candidates=inputs.candidates if candidates is None else candidates
    ego_box=rear_axle_to_box(candidates,inputs.ego[:,-1,5:7][:,None,None],inputs.ego_rear_axle_to_center[:,None,None])
    agent_box=torch.cat((future_agents[...,:3],future_agents[...,5:7]),-1)
    events=(sat_clearance(ego_box[:,:,:,None,:],agent_box[:,None])<=0) & future_valid[:,None]
    return events.any(-2).any(-1).float(),events.any(-2),events


def box_polygon(pose, length, width, rear_axle_offset=0.0):
    from shapely.geometry import Polygon
    x,y,yaw=map(float,pose)
    c,s=np.cos(yaw),np.sin(yaw)
    points=np.array([[length/2,width/2],[length/2,-width/2],[-length/2,-width/2],[-length/2,width/2]])
    return Polygon(points @ np.array([[c,s],[-s,c]]) + [x+rear_axle_offset*c,y+rear_axle_offset*s])


def signed_containment_margin(footprint, allowed):
    if allowed.is_empty:return -1e3
    if allowed.covers(footprint):return float(footprint.distance(allowed.boundary))
    # Negative penetration proxy, not an exact signed distance for nonconvex polygons.
    outside=footprint.difference(allowed)
    return -max(float(np.sqrt(outside.area)),1e-3)


def map_rule_margins(candidates, length_width, offset, map_tokens):
    """Map tokens contain fixed geometry and CURRENT control state only.

    lane/drivable: geometry must already be unioned allowed polygons.
    stop/signal: prohibited conflict polygon, active only when required by current
    control context. Stop-line crossing is tracked in sequence; a prior near-line
    stop satisfies stop tokens. Returned validity preserves missing observations.
    """
    from shapely.geometry import Point
    m,h,_=candidates.shape
    margins=np.zeros((m,h,len(map_tokens),4),np.float32)
    valid=np.zeros_like(margins,dtype=bool)
    for j,token in enumerate(map_tokens):
        kind=token['kind']; r={'lane':0,'drivable':1,'stop':2,'signal':3}[kind]
        if not token.get('available',True) or not token.get('rule_enabled',True):continue
        active=token.get('prohibited',False)
        if kind=='signal' and not token.get('signal_valid',False):continue
        if kind=='signal' and not active:
            # Known permissive signal is a valid safe condition.
            margins[:,:,j,r]=100.0; valid[:,:,j,r]=True; continue
        for i in range(m):
            stopped=token.get('initially_stopped',False)
            previous=np.zeros(3)
            for k,pose in enumerate(candidates[i]):
                footprint=box_polygon(pose,*length_width,offset)
                if kind in ('lane','drivable'):
                    margin=signed_containment_margin(footprint,token['geometry'])
                elif kind=='signal':
                    overlap=footprint.intersection(token['geometry'])
                    margin=-max(np.sqrt(overlap.area),1e-3) if not overlap.is_empty else footprint.distance(token['geometry'])
                else:
                    speed=np.linalg.norm(pose[:2]-previous[:2])/token.get('dt',0.5)
                    line=token['stop_line']
                    if speed<token.get('stop_speed',0.3) and footprint.distance(line)<token.get('stop_distance',2.0):stopped=True
                    crossed=token['geometry'].intersects(footprint)
                    margin=-1.0 if crossed and not stopped and active else 100.0
                margins[i,k,j,r]=margin;valid[i,k,j,r]=True;previous=pose
    return margins,valid
