"""Identical continuous strip contacts on a packed conservative graph."""
import torch

def packed_path_contact(paths,info,leader_xy,leader_yaw,leader_length,neighbors):
    """Every actor against the conservatively retained neighbors: same 32-strip test.

    Non-overlapping complete static path envelopes are excluded before rollout.
    No nearest-K truncation; neighbors are in original actor order.
    """
    ids=torch.cat((torch.arange(0,256,8,device=paths.device),torch.full((1,),255,device=paths.device,dtype=torch.long)))
    p=paths[:,ids,:2];a=p[:,:-1];b=p[:,1:];d=b-a;length=torch.linalg.vector_norm(d,dim=-1).clamp_min(1e-8);t=d/length[...,None]
    rel=leader_xy[:,:,:,None]-a[None,:,None];tx=t[None,:,None,:,0];ty=t[None,:,None,:,1]
    cx=rel[...,0]*tx+rel[...,1]*ty;cy=-rel[...,0]*ty+rel[...,1]*tx
    c=leader_yaw.cos()[...,None];s=leader_yaw.sin()[...,None];ca=c*tx+s*ty;sa=-c*ty+s*tx
    hl=leader_length[...,None]/2;hw=info[neighbors,4][None,:,:,None]/2;radius=info[None,:,None,None,4]/2
    xs=[];ys=[]
    for sx,sy in [(1,1),(1,-1),(-1,-1),(-1,1)]:xs.append(cx+sx*hl*ca-sy*hw*sa);ys.append(cy+sx*hl*sa+sy*hw*ca)
    low=torch.full_like(cx,float('inf'));high=-low
    for i in range(4):
        x,y=xs[i],ys[i];inside=y.abs()<=radius;low=torch.minimum(low,x.masked_fill(~inside,float('inf')));high=torch.maximum(high,x.masked_fill(~inside,-float('inf')))
        dx=xs[(i+1)%4]-x;dy=ys[(i+1)%4]-y;safe=torch.where(dy.abs()<1e-9,torch.ones_like(dy),dy)
        for sign in [-1.,1.]:
            u=(sign*radius-y)/safe;hit=(dy.abs()>=1e-9)&(u>=0)&(u<=1);v=x+u*dx
            low=torch.minimum(low,v.masked_fill(~hit,float('inf')));high=torch.maximum(high,v.masked_fill(~hit,-float('inf')))
    sl=length[None,:,None];valid=(low<=sl)&(high>=0)&(low<=high);u=high.clamp_min(0).minimum(sl)/sl
    progress=(ids[:-1]+u*(ids[1:]-ids[:-1]))*info[None,:,None,None,1]/255
    return progress.masked_fill(~valid,-float('inf')).amax(-1)
