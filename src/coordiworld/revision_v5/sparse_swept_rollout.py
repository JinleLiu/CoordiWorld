"""Current-only piecewise curved footprints and within-step interaction.

The footprint is a union of four flat strip rectangles plus the vehicle box.
This approximates the native curved buffer; it does not reproduce round joins
or world-coordinate nearest-neighbour tie ordering.
"""
import math
import torch
from .fast_rail_v3 import interpolate
from .rail_contact import box_path_contact
from .rail_geometry_capture import rectangle_gap, pair_path_contact


from .sparse_swept_step import step as _unfused_step

def _transition(paths,info,ego,p,v,initial,first,alive,contact,neighbors,neighbor_valid,pieces=4,iterations=2):
    q=_unfused_step(paths,info,ego,p,v,initial,first,alive,None,contact,neighbors,neighbor_valid,pieces)
    for _ in range(iterations):
        nxt=torch.stack((p+.1*v*alive,torch.where(alive,(v+.1*q['acceleration']).clamp_min(0),v)),-1)
        q=_unfused_step(paths,info,ego,p,v,initial,first,alive,nxt,contact,neighbors,neighbor_valid,pieces)
    return q

_transition_compiled=torch.compile(_transition,fullgraph=True,dynamic=False)

def rollout(paths,info,ego,initial,neighbors,neighbor_valid,iterations=2,pieces=4,trace=False):
    m,n=ego.shape[0],len(info)
    if not n:return paths.new_zeros(m,8,0,7),[]
    p=info[:,0][None].expand(m,-1).clone();v=info[:,2][None].expand(m,-1).clone();end=info[:,1]
    alive=torch.ones_like(v,dtype=torch.bool);contact=box_path_contact(paths,info,ego);outputs=[];traces=[]
    for k in range(40):
        pose=interpolate(paths,p,end);alive=alive&(torch.linalg.vector_norm(pose[...,:2]-ego[:,k,None,:2],dim=-1)<=100)
        q=_transition_compiled(paths,info,ego[:,k],p,v,initial,0 if k==0 else 1,alive,contact[:,k],neighbors,neighbor_valid,pieces,iterations)
        if trace:traces.append({**q,'progress':p,'speed':v,'alive':alive})
        p=p+.1*v*alive;v=torch.where(alive,(v+.1*q['acceleration']).clamp_min(0),v)
        if (k+1)%5==0:
            pose=interpolate(paths,p,end);velocity=v[...,None]*torch.stack((pose[...,2].cos(),pose[...,2].sin()),-1)
            outputs.append(torch.cat((pose,velocity,info[None,:,3:5].expand(m,-1,-1)),-1))
    return torch.stack(outputs,1),traces
