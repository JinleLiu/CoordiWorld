"""Exact box-versus-piecewise-flat-strip contact, all in current local frame.

The path is still the fixed 256-sample current segment. Internal round joins
of the native Shapely buffer are not represented; this approximation is
explicit and substantially tighter than expanding an ego box in both axes.
"""
import torch


def box_path_contact(paths,info,ego):
    a=paths[:,:-1,:2];b=paths[:,1:,:2];delta=b-a
    seglen=torch.linalg.vector_norm(delta,dim=-1).clamp_min(1e-8)
    tangent=delta/seglen[...,None]
    rel=ego[:,:,None,None,:2]-a[None,None]
    tx=tangent[None,None,:,:,0];ty=tangent[None,None,:,:,1]
    cx=rel[...,0]*tx+rel[...,1]*ty;cy=-rel[...,0]*ty+rel[...,1]*tx
    ec=ego[:,:,2].cos()[:,:,None,None];es=ego[:,:,2].sin()[:,:,None,None]
    ca=ec*tx+es*ty;sa=-ec*ty+es*tx
    radius=info[None,None,:,None,4]/2
    xs=[];ys=[]
    for lx,ly in [(2.588,1.1485),(2.588,-1.1485),(-2.588,-1.1485),(-2.588,1.1485)]:
        xs.append(cx+lx*ca-ly*sa);ys.append(cy+lx*sa+ly*ca)
    low=torch.full_like(cx,float('inf'));high=-low
    for i in range(4):
        x,y=xs[i],ys[i];inside=y.abs()<=radius
        low=torch.minimum(low,x.masked_fill(~inside,float('inf')))
        high=torch.maximum(high,x.masked_fill(~inside,-float('inf')))
        dx=xs[(i+1)%4]-x;dy=ys[(i+1)%4]-y;safe=torch.where(dy.abs()<1e-9,torch.ones_like(dy),dy)
        for sign in [-1.,1.]:
            t=(sign*radius-y)/safe;hit=(dy.abs()>=1e-9)&(t>=0)&(t<=1);v=x+t*dx
            low=torch.minimum(low,v.masked_fill(~hit,float('inf')))
            high=torch.maximum(high,v.masked_fill(~hit,-float('inf')))
    length=seglen[None,None];valid=(low<=length)&(high>=0)&(low<=high)
    u=(high.clamp_min(0).minimum(length)/length).clamp(0,1)
    ids=torch.arange(255,device=paths.device,dtype=paths.dtype)
    progress=(ids+u)*info[None,None,:,None,1]/255
    return progress.masked_fill(~valid,-float('inf')).amax(-1)
