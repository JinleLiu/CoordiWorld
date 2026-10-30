"""Current-input candidate features from explicit world states, with no future labels.

SAT margins are geometric quantities, not calibrated event probabilities. All
current entities remain in the input mask. No averaging over entity count is
used, so sufficiently distant noncontributing entities do not dilute risk.
"""
import torch
from ..geometry import rear_axle_to_box, sat_clearance, wrap_angle


def candidate_features(scene, executed, rule_margins, rule_valid, world):
    """One scene: world[M,8,N,7], executed[M,8,3]; returns[M,188]."""
    assert world.ndim==4 and world.shape[:2]==executed.shape[:2]
    m,t,n,_=world.shape;assert t==8
    init=scene.ego[0,-1,:2].expand(m,1,2)
    velocity=torch.diff(torch.cat((init,executed[...,:2]),1),dim=1)/.5
    acceleration=torch.diff(torch.cat((scene.ego[0,-1,3:5].expand(m,1,2),velocity),1),dim=1)/.5
    motion=torch.cat((executed[...,:2]/50,executed[...,2:].sin(),executed[...,2:].cos(),velocity/20,acceleration/10),-1).flatten(1)
    box=rear_axle_to_box(executed,scene.ego[0,-1,5:7],scene.ego_rear_axle_to_center[0])
    valid=scene.agent_valid[0,-1]
    if n:
        agents=torch.cat((world[...,:3],world[...,5:7]),-1)
        gap=sat_clearance(box[:,:,None],agents).masked_fill(~valid[None,None],50).clamp(-10,50)
        if n<4:gap=torch.cat((gap,gap.new_full((m,t,4-n),50)),dim=-1)
        nearest=gap.topk(4,dim=-1,largest=False).values/10
        true_gap=gap[...,:n]
        overlap=((true_gap<=0)&valid[None,None]).sum(-1).clamp_max(10).float()/10
        proximity=((1-true_gap/5).clamp(0,1)*valid[None,None]).sum(-1).clamp_max(10)/10
        relative=world[...,:2]-box[:,:,None,:2];distance=relative.norm(dim=-1).clamp_min(.01)
        rv=world[...,3:5]-velocity[:,:,None]
        closing=-(relative*rv).sum(-1)/distance
        ttc=torch.where((closing>.1)&valid[None,None],distance/closing.clamp_min(.1),10).clamp(0,10).amin(-1)/10
        dynamic=torch.cat((nearest,overlap[...,None],proximity[...,None],ttc[...,None]),-1).flatten(1)
    else:
        dynamic=world.new_tensor([5.,5.,5.,5.,0.,0.,1.]).repeat(m,t).reshape(m,-1)
    available=rule_valid&scene.map_valid[0,None,None,:,None]
    margin=rule_margins.masked_fill(~available,100).amin(2).clamp(-100,100)/10
    known=available.any(2).to(world.dtype)
    rules=torch.cat((margin,known),-1).flatten(1)
    speed=scene.ego[0,-1,3:5].norm()/20
    dv=(scene.ego[0,-1,3:5]-scene.ego[0,-2,3:5])/.5/10
    yaw=wrap_angle(scene.ego[0,-1,2]-scene.ego[0,-2,2])/.5
    common=torch.cat((speed.reshape(1),dv,yaw.reshape(1))).expand(m,4)
    features=torch.cat((motion,dynamic,rules,common),-1)
    assert features.shape==(m,188) and torch.isfinite(features).all()
    return features


def component_utility(c):
    return c[...,:4].prod(-1)*(c[...,4:]*c.new_tensor([5.,5.,2.,2.])).sum(-1)/14
