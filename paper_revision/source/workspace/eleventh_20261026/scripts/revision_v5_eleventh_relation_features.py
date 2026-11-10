"""Current-only actor-relation features; proxies, not native event labels.

Keeps actor identity when excluding initial contacts. Front/rear and stopped
relations are explicit; lateral fault still needs learned road context.
"""
import torch
from coordiworld.geometry import rear_axle_to_box, sat_clearance


def relation_features(scene, executed, world):
    m,t,n,_=world.shape
    box=rear_axle_to_box(executed,scene.ego[0,-1,5:7],scene.ego_rear_axle_to_center[0])
    velocity=torch.diff(torch.cat((scene.ego[0,-1,:2].expand(m,1,2),executed[...,:2]),1),dim=1)/.5
    speed=velocity.norm(dim=-1)
    out=world.new_zeros(m,t,12);out[...,:6]=5;out[...,10]=speed/20
    if not n or not scene.agent_valid[0,-1].any():return out.flatten(1)
    valid=scene.agent_valid[0,-1][None,None].expand(m,t,n)
    cur=scene.agents[0,-1]
    initbox=rear_axle_to_box(scene.ego[0,-1,:3],scene.ego[0,-1,5:7],scene.ego_rear_axle_to_center[0])
    initial_contact=sat_clearance(initbox,torch.cat((cur[:,:3],cur[:,5:7]),-1))<=0
    active=valid & ~initial_contact[None,None]
    gap=sat_clearance(box[:,:,None],torch.cat((world[...,:3],world[...,5:7]),-1)).clamp(-10,50)
    rel=world[...,:2]-box[:,:,None,:2]
    heading=torch.stack((executed[...,2].cos(),executed[...,2].sin()),-1)
    lon=(rel*heading[:,:,None]).sum(-1)
    lat=rel[...,1]*heading[:,:,None,0]-rel[...,0]*heading[:,:,None,1]
    distance=rel.norm(dim=-1).clamp_min(.01)
    front=lon/distance>.5;rear=lon/distance<-.5;side=~(front|rear)
    stopped=cur[:,3:5].norm(dim=-1)<=.05
    groups=[active&front,active&stopped[None,None],active&rear,active&side,active,active&(front|stopped[None,None])&(speed[:,:,None]>.05)]
    for k,mask in enumerate(groups):out[...,k]=gap.masked_fill(~mask,50).amin(-1)/10
    closest=gap.masked_fill(~active,50).argmin(-1)
    def get(x):return x.gather(-1,closest[...,None]).squeeze(-1)
    have=active.any(-1)
    closing=-(rel*(world[...,3:5]-velocity[:,:,None])).sum(-1)/distance
    out[...,6]=torch.where(have,get(lon).clamp(-50,50)/50,0)
    out[...,7]=torch.where(have,get(lat).clamp(-50,50)/50,0)
    out[...,8]=torch.where(have,get(closing).clamp(-30,30)/30,0)
    actor=scene.agent_class[0][None,None].expand(m,t,n)
    out[...,9]=torch.where(have,get(actor).float()/10,0)
    out[...,11]=((gap<=0)&active&side).any(-1).float()
    assert out.shape==(m,t,12) and torch.isfinite(out).all()
    return out.flatten(1)
