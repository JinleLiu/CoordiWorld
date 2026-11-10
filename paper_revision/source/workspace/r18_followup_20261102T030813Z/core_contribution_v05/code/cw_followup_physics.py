# -*- coding: utf-8 -*-
"""Head-independent observed-box/CV diagnostics. No future observations."""
import numpy as np

def rotation(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.stack([np.stack([c,-s],-1),np.stack([s,c],-1)],-2)

def local_native(native, frame):
    p=np.array(native[...,:3],dtype=float,copy=True)
    yaw=float(frame['body_pose_math'][2]);origin=np.asarray(frame['ego2global_translation'][:2])
    p[...,:2]=(p[...,:2]-origin)@rotation(yaw)
    p[...,2]=np.arctan2(np.sin(p[...,2]-yaw),np.cos(p[...,2]-yaw))
    return p

def cv_box_check(path, frame, dt=.1, clearance=.1, stationary_speed=0.):
    """SAT at samples; velocities in current rear frame, fixed actor heading.

    A positive separating-axis value is a box separation witness at that sample,
    not continuous Euclidean clearance or a dynamic-safety certificate.
    """
    path=np.asarray(path,float); boxes=np.asarray(frame['anns']['gt_boxes'],float).reshape(-1,7)
    velocity=np.array(frame['anns']['gt_velocity_3d'],dtype=float,copy=True).reshape(-1,3)
    velocity[np.linalg.norm(velocity[:,:2],axis=1)<=stationary_speed]=0.
    if not len(boxes):
        return dict(min_separation=1000.,first_below_clearance_s=None,actor=None,cv_clear=True)
    geo=frame['rear_geometry'];extent=np.asarray(geo['box_extent'][:2]);offset=np.array([geo['box_offset'][0],-geo['box_offset'][1]])-np.asarray(geo['rear_offset_math'])
    r=rotation(path[:,2]); rb=rotation(boxes[:,6]);center=path[:,:2]+r@offset
    times=np.arange(len(path))*dt; targets=boxes[None,:,:2]+times[:,None,None]*velocity[None,:,:2]
    axes=np.concatenate([np.broadcast_to(r.transpose(0,2,1)[:,None],(len(path),len(boxes),2,2)),np.broadcast_to(rb.transpose(0,2,1)[None],(len(path),len(boxes),2,2))],axis=2)
    distance=abs(np.einsum('saki,sai->sak',axes,targets-center[:,None]))
    radii=abs(np.einsum('saki,sij->sakj',axes,r))@extent+np.einsum('sakj,aj->sak',abs(np.einsum('saki,aij->sakj',axes,rb)),boxes[:,3:5]/2)
    separation=(distance-radii).max(-1);idx=np.unravel_index(separation.argmin(),separation.shape)
    hits=np.flatnonzero((separation<clearance).any(1))
    return dict(min_separation=float(separation[idx]),first_below_clearance_s=None if not len(hits) else float(times[hits[0]]),actor=str(frame['anns']['track_tokens'][idx[1]]),cv_clear=not bool(len(hits)))

def simple_rank(native, frame, mask, route_progress=None, stationary_speed=0.):
    """Fixed lexicographic CV-clear/TTC/forward-progress/heading ranking.

    Only deployment-visible current observations and deterministic predicted ego
    execution are used. No CW head or future labels. Caller supplies the exact
    L0 eligibility mask, or the independent L1 physical mask.
    """
    paths=local_native(native,frame);rows=[]
    for i,path in enumerate(paths):
        cv=cv_box_check(path,frame,stationary_speed=stationary_speed)
        progress=float(path[-1,0]) if route_progress is None else float(route_progress[i]); heading=float(abs(np.arctan2(np.sin(path[-1,2]),np.cos(path[-1,2]))))
        ttc=4.1 if cv['first_below_clearance_s'] is None else cv['first_below_clearance_s']
        # TTC separates predicted collisions; among clear paths prefer progress.
        key=(int(cv['cv_clear']),ttc,progress,-heading,-i)
        rows.append(dict(index=i,valid=bool(mask[i]),progress_forward_m=float(path[-1,0]),rank_progress_m=progress,progress_source='body_forward' if route_progress is None else 'deterministic_native_PDM_route_projection',terminal_heading_abs=heading,rank_key=list(key),**cv))
    valid=np.flatnonzero(mask)
    if not len(valid): raise ValueError('No eligible candidate: caller must record a physical emergency, not silently argmax an empty set')
    choice=max((int(i) for i in valid),key=lambda i:tuple(rows[i]['rank_key']))
    return choice,rows
