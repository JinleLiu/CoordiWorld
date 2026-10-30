"""Batched geometric primitives; no trained parameters or teacher inputs."""
import torch


def rectangle_gap(lon,lat,ca,sa,own_half_length,own_half_width,lead_half_length,lead_half_width):
    """Exact Euclidean distance between two oriented rectangles in own axes."""
    best=torch.full_like(lon,float('inf'))
    for sx,sy in [(1,1),(1,-1),(-1,-1),(-1,1)]:
        x=lon+sx*lead_half_length*ca-sy*lead_half_width*sa
        y=lat+sx*lead_half_length*sa+sy*lead_half_width*ca
        d=(x.abs()-own_half_length).clamp_min(0).square()+(y.abs()-own_half_width).clamp_min(0).square()
        best=torch.minimum(best,d)
        rx=sx*own_half_length-lon;ry=sy*own_half_width-lat
        x=rx*ca+ry*sa;y=-rx*sa+ry*ca
        d=(x.abs()-lead_half_length).clamp_min(0).square()+(y.abs()-lead_half_width).clamp_min(0).square()
        best=torch.minimum(best,d)
    overlap=(lon.abs()<=own_half_length+lead_half_length*ca.abs()+lead_half_width*sa.abs())&(lat.abs()<=own_half_width+lead_half_length*sa.abs()+lead_half_width*ca.abs())
    overlap=overlap&((lon*ca+lat*sa).abs()<=lead_half_length+own_half_length*ca.abs()+own_half_width*sa.abs())&((-lon*sa+lat*ca).abs()<=lead_half_width+own_half_length*sa.abs()+own_half_width*ca.abs())
    return torch.where(overlap,torch.zeros_like(best),best).clamp_min(0).sqrt()


def pair_path_contact(paths,info,leader_xy,leader_yaw,leader_length):
    """Every actor against every future path: coarse continuous 32-strip test.

    All current active actors are retained. Only path tessellation is coarser;
    no nearest-neighbor truncation. Rectangle projected occupancies remain an
    approximation to curved native occupancies and are documented as such.
    """
    ids=torch.cat((torch.arange(0,256,8,device=paths.device),torch.full((1,),255,device=paths.device,dtype=torch.long)))
    p=paths[:,ids,:2];a=p[:,:-1];b=p[:,1:];d=b-a;length=torch.linalg.vector_norm(d,dim=-1).clamp_min(1e-8);t=d/length[...,None]
    rel=leader_xy[:,:,:,None]-a[None,:,None];tx=t[None,:,None,:,0];ty=t[None,:,None,:,1]
    cx=rel[...,0]*tx+rel[...,1]*ty;cy=-rel[...,0]*ty+rel[...,1]*tx
    c=leader_yaw.cos()[...,None];s=leader_yaw.sin()[...,None];ca=c*tx+s*ty;sa=-c*ty+s*tx
    hl=leader_length[...,None]/2;hw=info[None,None,:,None,4]/2;radius=info[None,:,None,None,4]/2
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
