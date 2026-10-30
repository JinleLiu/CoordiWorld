"""Current-only piecewise curved footprints and within-step interaction.

The footprint is a union of four flat strip rectangles plus the vehicle box.
This approximates the native curved buffer; it does not reproduce round joins
or world-coordinate nearest-neighbour tie ordering.
"""
import math
import torch
from .fast_rail_v3 import interpolate
from .rail_contact import box_path_contact
from .rail_geometry_capture import rectangle_gap
from .packed_path_contact import packed_path_contact


def footprint(paths, info, p, v, initial, initial_mask, pieces=4):
    m,n=p.shape;end=info[:,1];length=info[:,3];width=info[:,4]
    start=(p-length[None]/2).clamp_min(0).minimum(end[None])
    stop=(p+length[None]/2+1.5*v).clamp_min(0).minimum(end[None])
    u=torch.linspace(0,1,pieces+1,device=p.device,dtype=p.dtype)
    progress=start[:,None,:]+u[None,:,None]*(stop-start)[:,None,:]
    poses=interpolate(paths,progress.reshape(m*(pieces+1),n),end).reshape(m,pieces+1,n,3).permute(0,2,1,3)
    a,b=poses[:,:,:-1,:2],poses[:,:,1:,:2];d=b-a
    xy=.5*(a+b);heading=torch.atan2(d[...,1],d[...,0]);half=torch.linalg.vector_norm(d,dim=-1)/2
    body=interpolate(paths,p,end);body=torch.where(initial_mask[...,None],initial[None],body)
    xy=torch.cat((xy,body[:,:,None,:2]),2);heading=torch.cat((heading,body[:,:,None,2]),2)
    half=torch.cat((half,length[None,:,None].expand(m,-1,1)/2),2)
    valid=torch.cat(((half[:,:,:pieces]>1e-7)&~initial_mask[:,:,None],torch.ones(m,n,1,device=p.device,dtype=torch.bool)),2)
    return {'xy':xy,'yaw':heading,'hl':half,'hw':width[None,:,None].expand_as(half)/2,'valid':valid}


def step(paths,info,ego,p,v,initial,k,alive,ordered,ego_contact,neighbors,neighbor_valid,pieces=4):
    m,n=p.shape;device=p.device;count_neighbors=neighbors.shape[1];earlier=neighbors<torch.arange(n,device=device)[:,None]
    curpose=interpolate(paths,p,info[:,1]);curinit=torch.full_like(alive,k==0)
    own=footprint(paths,info,p,v,initial,curinit,pieces)
    if ordered is None:
        nxt=own;newpose=curpose;newv=v
    else:
        nxt=footprint(paths,info,ordered[...,0],ordered[...,1],initial,torch.zeros_like(alive),pieces)
        newpose=interpolate(paths,ordered[...,0],info[:,1]);newv=ordered[...,1]
    lead={}
    for name,value in own.items():
        if name=='xy':lead[name]=torch.where(earlier[None,:,:,None,None],nxt[name][:,neighbors],value[:,neighbors])
        else:lead[name]=torch.where(earlier[None,:,:,None],nxt[name][:,neighbors],value[:,neighbors])
    # Distances between all rectangle components, retaining all current actors.
    rel=lead['xy'][:,:,:,None,:,:]-own['xy'][:,:,None,:,None,:]
    yaw=own['yaw'][:,:,None,:,None];c,s=yaw.cos(),yaw.sin()
    lon=rel[...,0]*c+rel[...,1]*s;lat=-rel[...,0]*s+rel[...,1]*c
    angle=lead['yaw'][:,:,:,None,:]-yaw
    gaps=rectangle_gap(lon,lat,angle.cos(),angle.sin(),own['hl'][:,:,None,:,None],own['hw'][:,:,None,:,None],lead['hl'][:,:,:,None,:],lead['hw'][:,:,:,None,:])
    valid=own['valid'][:,:,None,:,None]&lead['valid'][:,:,:,None,:]
    gaps=gaps.masked_fill(~valid,float('inf')).amin((-1,-2))
    count=pieces+1
    # Test the same union against each actor's remaining path, rather than a
    # single straight extrapolated leader rectangle.
    lx=lead['xy'].permute(0,3,1,2,4).reshape(m*count,n,count_neighbors,2)
    ly=lead['yaw'].permute(0,3,1,2).reshape(m*count,n,count_neighbors)
    ll=(2*lead['hl']).permute(0,3,1,2).reshape(m*count,n,count_neighbors)
    contact=packed_path_contact(paths,info,lx,ly,ll,neighbors).reshape(m,count,n,count_neighbors).permute(0,2,3,1)
    eligible=((contact>=p[:,:,None,None])&lead['valid']).any(-1)&neighbor_valid[None]&alive[:,neighbors]
    gaps=gaps.masked_fill(~eligible,float('inf'));nearest,packed_index=gaps.min(-1);index=neighbors[None].expand(m,-1,-1).gather(2,packed_index[...,None]).squeeze(-1)
    lv=torch.where(earlier[None],newv[:,neighbors],v[:,neighbors]);lyaw=torch.where(earlier[None],newpose[:,neighbors,2],curpose[:,neighbors,2]);lspeed=lv*(lyaw-curpose[:,:,None,2]).cos();lead_speed=lspeed.gather(2,packed_index[...,None]).squeeze(-1)
    er=ego[:,None,None,:2]-own['xy'];c=own['yaw'].cos();s=own['yaw'].sin();el=er[...,0]*c+er[...,1]*s;et=-er[...,0]*s+er[...,1]*c;angle=ego[:,None,None,2]-own['yaw']
    egap=rectangle_gap(el,et,angle.cos(),angle.sin(),own['hl'],own['hw'],2.588,1.1485).masked_fill(~own['valid'],float('inf')).amin(-1)
    egap=egap.masked_fill(ego_contact<p,float('inf'));useego=egap<=nearest;nearest=torch.minimum(egap,nearest);lead_speed=torch.where(useego,ego[:,None,3]*(ego[:,None,2]-curpose[...,2]).cos(),lead_speed)
    free=~torch.isfinite(nearest);gap=torch.where(free,info[None,:,1]-p-info[None,:,3]/2,nearest);lead_speed=torch.where(free,0.,lead_speed);leader=torch.where(free,-2,torch.where(useego,-1,index))
    desired=1+v*1.5+v*(v-lead_speed)/(2*math.sqrt(2.));acc=(1-(v/info[None,:,5].clamp_min(.1)).pow(4)-(desired/gap.clamp_min(1.)).square()).clamp(-2,1)
    return {'acceleration':acc,'leader':leader,'gap':gap,'lead_speed':lead_speed}

